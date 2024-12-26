import logging
from typing import Annotated, Union 
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
    multimodal,
    utils,
)
import aiohttp
import asyncio
from livekit.agents.llm import ChatMessage
from livekit.agents.multimodal.multimodal_agent import EventTypes
from livekit.agents.pipeline import VoicePipelineAgent, AgentCallContext
from livekit.plugins import openai, deepgram, silero, elevenlabs, turn_detector

OPENAI_LLM_INPUT_PRICE = 2.50 / (10**6)  # $2.50 per million tokens
OPENAI_LLM_OUTPUT_PRICE = 10 / (10**6)  # $10 per million tokens
ELVENLABS_TTS_PRICE = 0.0003  # 0.0003$ per character
DEEPGRAM_STT_PRICE = 0.0043  # $0.0043 per minute

class AssistantFnc(llm.FunctionContext):
    # the llm.ai_callable decorator marks this function as a tool available to the LLM
    # by default, it'll use the docstring as the function's description
    @llm.ai_callable()
    async def get_weather(
        self,
        # by using the Annotated type, arg description and type are available to the LLM
        location: Annotated[
            str, llm.TypeInfo(description="The location to get the weather for")
        ],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        logger.info(f"getting weather for {location}")
        call_ctx = AgentCallContext.get_current()
        await call_ctx.agent.say("Ummm....")

        url = f"https://wttr.in/{location}?format=%C+%t"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_data = await response.text()
                    # response from the function call is returned to the LLM
                    # as a tool response. The LLM's response will include this data
                    return f"The weather in {location} is {weather_data}."
                else:
                    raise f"Failed to get weather data, status code: {response.status}"

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
            "You were created as a demo to showcase the capabilities of LiveKit's agents framework."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")
    fnc_ctx = AssistantFnc()

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        turn_detector=turn_detector.EOUModel(),
        stt=deepgram.STT(model="nova-2-phonecall"),
        llm=openai.LLM(model="gpt-4o"),
        tts=elevenlabs.TTS(model="eleven_flash_v2"),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )

    usage_collector = metrics.UsageCollector()


    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_session_cost():
        summary = usage_collector.get_summary()
        llm_cost = (
            summary.llm_prompt_tokens * OPENAI_LLM_INPUT_PRICE
            + summary.llm_completion_tokens * OPENAI_LLM_OUTPUT_PRICE
        )
        tts_cost = summary.tts_characters_count * ELVENLABS_TTS_PRICE
        stt_cost = summary.stt_audio_duration * DEEPGRAM_STT_PRICE / 60

        total_cost = llm_cost + tts_cost + stt_cost

        logger.info(
            f"Total cost: ${total_cost:.4f} (LLM: ${llm_cost:.4f}, TTS: ${tts_cost:.4f}, STT: ${stt_cost:.4f})"
        )

    ctx.add_shutdown_callback(log_session_cost)
    agent.start(ctx.room, participant)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
