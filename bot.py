import os
import discord
import logging
import asyncio

from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent

PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Remove default help command since we show instructions by default
bot.remove_command('help')

# Import the Mistral agent from the agent.py file
agent = MistralAgent()

# Bot instructions
INSTRUCTIONS = """
🌟 **Welcome to Finna - Your Financial Journaling Workbook!** 📈

Finna helps you track and analyze your investment ideas. Here's how to use it:

📝 **Simple Stock Journaling:**
• Just type any stock ticker (e.g., `AAPL`, `TSLA`, `GOOGL`)
• Finna remembers your previous entries and tracks price changes
• Perfect for journaling your trading intuitions

📊 **Portfolio Analysis:**
• Input stocks with quantities using `TICKER:QUANTITY` format
• Example: `AAPL:10 MSFT:5 GOOGL:3`
• Get insights on:
  - Portfolio distribution
  - Risk analysis
  - Long-term investment considerations

🛠 **Commands:**
• `!help` - Show these instructions
• `!clear [number]` - Clear messages (admin only)

Start journaling your investment ideas and let Finna help you make informed decisions! 💡
"""

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")


@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    logger.info(f"{bot.user} has connected to Discord!")
    
    # Clear messages and show instructions in all text channels the bot can see
    for guild in bot.guilds:
        for channel in guild.text_channels:
            try:
                # Check if bot has permission to manage messages
                if channel.permissions_for(guild.me).manage_messages:
                    # Purge all messages
                    await channel.purge(limit=None)
                    logger.info(f"Cleared messages in {channel.name}")
                    # Send welcome message with instructions
                    await channel.send(INSTRUCTIONS)
            except Exception as e:
                logger.error(f"Error clearing messages in {channel.name}: {e}")


@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_message
    """
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    if message.author.bot or message.content.startswith("!"):
        return

    # Process the message with the agent you wrote
    # Open up the agent.py file to customize the agent
    logger.info(f"Processing message from {message.author}: {message.content}")
    response = await agent.run(message)

    # Send the response back to the channel if there is one
    if response:
        await message.reply(response)


# Commands


# This example command is here to show you how to add commands to the bot.
# Run !ping with any number of arguments to see the command in action.
# Feel free to delete this if your project will not need commands.
@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")


@bot.command(name="clear", help="Clears messages from the channel. Usage: !clear [number] (default: all messages)")
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount: str = "all"):
    """
    Clears messages from the channel.
    Args:
        amount: Number of messages to clear or "all" for all messages
    """
    try:
        # Send confirmation message
        confirm_msg = await ctx.send(f"🗑️ Are you sure you want to clear {'all' if amount == 'all' else amount} messages?\n"
                                   f"React with ✅ to confirm or ❌ to cancel.")
        await confirm_msg.add_reaction("✅")
        await confirm_msg.add_reaction("❌")

        def check(reaction, user):
            return user == ctx.author and str(reaction.emoji) in ["✅", "❌"] and reaction.message == confirm_msg

        try:
            reaction, user = await bot.wait_for('reaction_add', timeout=30.0, check=check)
            
            if str(reaction.emoji) == "✅":
                # Delete the confirmation message
                await confirm_msg.delete()
                
                # Clear messages
                if amount.lower() == "all":
                    deleted = await ctx.channel.purge(limit=None)
                else:
                    try:
                        limit = int(amount) + 1  # +1 to include the command message
                        deleted = await ctx.channel.purge(limit=limit)
                    except ValueError:
                        error_msg = await ctx.send("❌ Please provide a valid number or 'all'")
                        await asyncio.sleep(5)
                        await error_msg.delete()
                        return
                
                # Send success message that deletes itself after 5 seconds
                success_msg = await ctx.send(f"✨ Successfully cleared {len(deleted)} messages!")
                await asyncio.sleep(5)
                await success_msg.delete()
            else:
                await confirm_msg.delete()
                cancel_msg = await ctx.send("❌ Message clearing cancelled.")
                await asyncio.sleep(5)
                await cancel_msg.delete()
                
        except asyncio.TimeoutError:
            await confirm_msg.delete()
            timeout_msg = await ctx.send("⏰ Clear command timed out.")
            await asyncio.sleep(5)
            await timeout_msg.delete()
            
    except discord.Forbidden:
        error_msg = await ctx.send("❌ I don't have permission to delete messages in this channel.")
        await asyncio.sleep(5)
        await error_msg.delete()
    except Exception as e:
        error_msg = await ctx.send(f"❌ An error occurred: {str(e)}")
        await asyncio.sleep(5)
        await error_msg.delete()

@clear.error
async def clear_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        msg = await ctx.send("❌ You don't have permission to clear messages!")
        await asyncio.sleep(5)
        await msg.delete()

@bot.event
async def on_guild_join(guild):
    """Called when the bot joins a new guild/server"""
    for channel in guild.text_channels:
        try:
            if channel.permissions_for(guild.me).send_messages:
                await channel.send(INSTRUCTIONS)
                break  # Send to first channel we have permission in
        except Exception as e:
            logger.error(f"Error sending welcome message in {channel.name}: {e}")

# Start the bot, connecting it to the gateway
bot.run(token)
