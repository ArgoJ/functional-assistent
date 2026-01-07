from typing import Any, Optional

def search_web(query: str) -> str:
    """
    Search public information.

    Args:
        query: query string
    """
    return "Public Result"

def device_control(device_name: str, location: str = "", action: str = "on", percentage: Optional[int] = None) -> str:
    """
    Control a smart home device with specified action.

    Args:
        device_name: The name of the device (e.g., "light", "heater").
        location: The location of the device (e.g., "living room").
        action: The action to perform (e.g., "on", "off", "dimm").
        percentage: Optional percentage for the action (e.g. for dimming).
    """
    return f"Turned on {device_name} in {location}"

def get_weather(location: str, date: str) -> str:
    """
    Get the weather forecast for a specific location and date.

    Args:
        location: The city or place.
        date: The date for the forecast (e.g., "now", "tomorrow", "2024-01-01").
    """
    return f"Weather in {location} on {date}: Sunny"

def create_timer(duration: str) -> str:
    """
    Create a timer in Google.

    Args:
        duration: The duration of the timer (e.g., "5 minutes").
    """
    return f"Timer set for {duration}"

def create_alarm(time: str, label: str = "") -> str:
    """
    Create an alarm.

    Args:
        time: The time for the alarm (e.g., "7:00 AM").
        label: A label for the alarm.
    """
    return f"Alarm set for {time}"

def play_music(genre: str = "", artist: str = "", album: str = "", playlist: str = "", song: str = "") -> str:
    """
    Play music on Spotify based on genre, artist, album, or playlist.

    Args:
        genre: The genre of music to play.
        artist: The artist to play.
        album: The album to play.
        playlist: The playlist to play.
        song: The song title to play.
    """
    return "Playing music on Spotify"