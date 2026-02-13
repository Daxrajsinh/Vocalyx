"""TTS WebSocket server - streams GPT responses and converts to speech using Soprano TTS."""

import os
import asyncio
import json
import websockets
import torch
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from soprano import SopranoTTS

from vocalyx._ui import info, success, error

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Soprano TTS once
print(info("Starting TTS server..."))
print(info("Loading model..."))
tts_model = SopranoTTS(
    backend="auto",
    device="auto",
    cache_size_mb=100,
    decoder_batch_size=1
)


class AudioWebSocketServer:
    def __init__(self, host="localhost", port=8013):
        self.host = host
        self.port = port

    async def handle_client(self, websocket):
        print(info("Client connected."))

        try:
            async for message in websocket:
                data = json.loads(message)

                if data.get("type") == "generate_speech":
                    prompt = data.get("prompt", "")
                    system_prompt = data.get("system_prompt", "You are a motivational assistant.")
                    model = data.get("model", "gpt-3.5-turbo")
                    print(info(f"Generating speech: {prompt[:50]}..."))
                    await self.stream_gpt_with_tts(websocket, prompt, system_prompt=system_prompt, model=model)

        except websockets.exceptions.ConnectionClosed:
            print(info("Client disconnected."))
        except Exception as e:
            print(error(f"Error: {e}"))
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))

    async def stream_gpt_with_tts(self, websocket, prompt, system_prompt="You are a motivational assistant.", model="gpt-3.5-turbo"):
        try:
            await websocket.send(json.dumps({
                "type": "generation_started"
            }))

            text_buffer = ""

            with client.responses.stream(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            ) as stream:

                for event in stream:
                    if event.type == "response.output_text.delta":
                        token = event.delta
                        text_buffer += token

                        # Speak on sentence boundary
                        if token in [".", "!", "?", "\n"] and text_buffer.strip():
                            await self.process_and_send_audio(
                                websocket, text_buffer.strip()
                            )
                            text_buffer = ""

                    elif event.type == "response.completed":
                        if text_buffer.strip():
                            await self.process_and_send_audio(
                                websocket, text_buffer.strip()
                            )

                        await websocket.send(json.dumps({
                            "type": "generation_completed"
                        }))
                        break

        except Exception as e:
            print(error(f"TTS generation failed: {e}"))
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"TTS generation failed: {str(e)}"
            }))

    async def process_and_send_audio(self, websocket, text):
        try:

            await websocket.send(json.dumps({
                "type": "text_chunk",
                "text": text
            }))

            audio_stream = tts_model.infer_stream(
                text,
                chunk_size=2
            )

            for chunk in audio_stream:
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu()

                # Match play_stream shape logic
                if chunk.dim() == 1:
                    chunk = chunk.unsqueeze(1)
                elif chunk.dim() == 2 and chunk.shape[0] == 1:
                    chunk = chunk.transpose(0, 1)

                chunk_np = chunk.numpy().astype(np.float32)

                await websocket.send(json.dumps({
                    "type": "audio_chunk",
                    "audio_data": chunk_np.tobytes().hex(),
                    "sample_rate": 32000,
                    "channels": 1,
                    "format": "f32"
                }))

        except Exception as e:
            print(error(f"Audio error: {e}"))
            raise

    async def start_server(self):
        print(success(f"TTS server ready on ws://{self.host}:{self.port}"))
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(info("Press Ctrl+C to stop."))
            await asyncio.Future()


async def main():
    server = AudioWebSocketServer()
    await server.start_server()


def cli():
    """Entry point for vocalyx-tts command."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(info("TTS server stopped."))


if __name__ == "__main__":
    cli()
