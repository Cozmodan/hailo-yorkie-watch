from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import hailo_vlm_server as server  # noqa: E402


class FakeArray:
    def __init__(self, label: str) -> None:
        self.label = label
        self.astype_calls: list[tuple[object, bool]] = []

    def astype(self, dtype: object, *, copy: bool = True) -> "FakeArray":
        self.astype_calls.append((dtype, copy))
        return self


class FakeNumpy:
    uint8 = "uint8"

    def __init__(self) -> None:
        self.frombuffer_calls: list[tuple[bytes, object]] = []

    def frombuffer(self, data: bytes, *, dtype: object) -> bytes:
        self.frombuffer_calls.append((data, dtype))
        return data


class FakeCV2:
    IMREAD_COLOR = 1
    COLOR_BGR2RGB = 2
    INTER_AREA = 3

    def __init__(self) -> None:
        self.resize_calls: list[tuple[FakeArray, tuple[int, int], int]] = []

    def imdecode(self, _buffer: object, flag: int) -> FakeArray:
        if flag != self.IMREAD_COLOR:
            raise AssertionError("unexpected imdecode flag")
        return FakeArray("bgr")

    def cvtColor(self, image: FakeArray, code: int) -> FakeArray:
        if image.label != "bgr" or code != self.COLOR_BGR2RGB:
            raise AssertionError("unexpected color conversion")
        return FakeArray("rgb")

    def resize(self, image: FakeArray, size: tuple[int, int], *, interpolation: int) -> FakeArray:
        self.resize_calls.append((image, size, interpolation))
        return FakeArray("resized")


class FakeVLM:
    def __init__(self, response: str = "Yes, there is a dog in the image.<|im_end|>") -> None:
        self.response = response
        self.clear_context_calls = 0
        self.generate_calls: list[tuple[object, object, int]] = []
        self.closed = False

    def clear_context(self) -> None:
        self.clear_context_calls += 1

    def generate(self, prompt: object, frames: object, *, max_tokens: int) -> str:
        self.generate_calls.append((prompt, frames, max_tokens))
        return self.response

    def close(self) -> None:
        self.closed = True


class FakeVDevice:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def encoded_image() -> str:
    return base64.b64encode(b"fake-jpeg").decode("ascii")


def server_config(*, unload_after_request: bool = True) -> server.ServerConfig:
    return server.ServerConfig(
        hef_path=server.DEFAULT_HEF,
        host="127.0.0.1",
        port=8010,
        max_tokens=12,
        optimize_memory=True,
        clear_context=True,
        unload_after_request=unload_after_request,
        model="Qwen2-VL-2B-Instruct",
    )


class HailoVLMServerTests(unittest.TestCase):
    def test_parse_chat_payload_builds_structured_prompt(self) -> None:
        request = server.parse_chat_payload(
            {
                "model": "Qwen2-VL-2B-Instruct",
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": "Is there a dog?",
                        "images": [encoded_image()],
                    }
                ],
            }
        )

        self.assertEqual(request.model, "Qwen2-VL-2B-Instruct")
        self.assertEqual(request.images, (encoded_image(),))
        self.assertEqual(
            request.prompt,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Is there a dog?"},
                    ],
                }
            ],
        )

    def test_parse_generate_payload_builds_image_placeholders(self) -> None:
        request = server.parse_generate_payload(
            {
                "model": "Qwen2-VL-2B-Instruct",
                "prompt": "Briefly describe the image.",
                "images": [encoded_image(), encoded_image()],
            }
        )

        self.assertEqual(len(request.images), 2)
        self.assertEqual(
            request.prompt[0]["content"],
            [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "Briefly describe the image."},
            ],
        )

    def test_clean_response_text_strips_hailo_special_tokens(self) -> None:
        self.assertEqual(
            server.clean_response_text(" Yes, there is a dog in the image.<|im_end|>\n"),
            "Yes, there is a dog in the image.",
        )

    def test_invalid_base64_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(server.RequestError, "valid base64"):
            server.decode_image_to_frame(
                "not-valid-base64",
                cv2_module=FakeCV2(),
                numpy_module=FakeNumpy(),
            )

    def test_decode_image_resizes_to_hailo_input_shape_and_uint8(self) -> None:
        cv2 = FakeCV2()
        np = FakeNumpy()

        frame = server.decode_image_to_frame(
            f"data:image/jpeg;base64,{encoded_image()}",
            cv2_module=cv2,
            numpy_module=np,
        )

        self.assertIsInstance(frame, FakeArray)
        self.assertEqual(cv2.resize_calls[0][1], (336, 336))
        self.assertEqual(frame.astype_calls, [("uint8", False)])
        self.assertEqual(np.frombuffer_calls[0], (b"fake-jpeg", "uint8"))

    def test_runtime_clears_context_and_generates_once(self) -> None:
        vlm = FakeVLM()
        runtime = server.HailoVLMRuntime(
            config=server_config(),
            vdevice=FakeVDevice(),
            vlm=vlm,
            cv2_module=FakeCV2(),
            numpy_module=FakeNumpy(),
        )
        request = server.parse_chat_payload(
            {
                "model": "Qwen2-VL-2B-Instruct",
                "messages": [{"role": "user", "content": "Is there a dog?", "images": [encoded_image()]}],
            }
        )

        text = runtime.generate(request)

        self.assertEqual(text, "Yes, there is a dog in the image.")
        self.assertEqual(vlm.clear_context_calls, 1)
        self.assertEqual(len(vlm.generate_calls), 1)
        self.assertEqual(vlm.generate_calls[0][2], 12)

    def test_unload_after_request_loads_and_closes_runtime_per_request(self) -> None:
        loaded_vlms: list[FakeVLM] = []
        loaded_vdevices: list[FakeVDevice] = []

        def load_runtime(config: server.ServerConfig) -> server.HailoVLMRuntime:
            vlm = FakeVLM("Temporary answer")
            vdevice = FakeVDevice()
            loaded_vlms.append(vlm)
            loaded_vdevices.append(vdevice)
            return server.HailoVLMRuntime(
                config=config,
                vdevice=vdevice,
                vlm=vlm,
                cv2_module=FakeCV2(),
                numpy_module=FakeNumpy(),
            )

        manager = server.HailoVLMRuntimeManager(
            config=server_config(unload_after_request=True),
            runtime_loader=load_runtime,
        )
        request = server.parse_generate_payload({"prompt": "Describe.", "images": [encoded_image()]})

        text = manager.generate(request)

        self.assertEqual(text, "Temporary answer")
        self.assertFalse(manager.loaded)
        self.assertEqual(len(loaded_vlms), 1)
        self.assertTrue(loaded_vlms[0].closed)
        self.assertTrue(loaded_vdevices[0].closed)

    def test_permanent_mode_reuses_loaded_runtime_until_closed(self) -> None:
        vlm = FakeVLM("Permanent answer")
        vdevice = FakeVDevice()
        runtime = server.HailoVLMRuntime(
            config=server_config(unload_after_request=False),
            vdevice=vdevice,
            vlm=vlm,
            cv2_module=FakeCV2(),
            numpy_module=FakeNumpy(),
        )
        manager = server.HailoVLMRuntimeManager(
            config=server_config(unload_after_request=False),
            runtime=runtime,
        )
        request = server.parse_generate_payload({"prompt": "Describe.", "images": [encoded_image()]})

        self.assertTrue(manager.loaded)
        self.assertEqual(manager.generate(request), "Permanent answer")
        self.assertTrue(manager.loaded)
        manager.close()

        self.assertFalse(manager.loaded)
        self.assertTrue(vlm.closed)
        self.assertTrue(vdevice.closed)

    def test_ollama_response_shape(self) -> None:
        response = server.ollama_response(model="Qwen2-VL-2B-Instruct", content="A dog is visible.")

        self.assertTrue(response["done"])
        self.assertEqual(response["message"], {"role": "assistant", "content": "A dog is visible."})
        self.assertEqual(response["response"], "A dog is visible.")


if __name__ == "__main__":
    unittest.main()
