def search_web(query: str) -> str:
    """
    Search public information.

    Args:
        query: query string
    """
    return "Public Result"

def openai_api_call(prompt: str) -> str:
    """
    Call OpenAI API.

    Args:
        prompt: prompt string
    """
    return "OpenAI Result"

def turn_on_device(device_name: str, location: str = "") -> str:
    """
    Turn on a smart home device.

    Args:
        device_name: The name of the device (e.g., "light", "heater").
        location: The location of the device (e.g., "living room").
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

def send_email(recipient: str, subject: str, body: str) -> str:
    """
    Send an email.

    Args:
        recipient: The email address of the recipient.
        subject: The subject of the email.
        body: The content of the email.
    """
    return f"Email sent to {recipient}"

def create_alarm(time: str, label: str = "") -> str:
    """
    Create an alarm.

    Args:
        time: The time for the alarm (e.g., "7:00 AM").
        label: A label for the alarm.
    """
    return f"Alarm set for {time}"

def play_music(genre: str = "", artist: str = "", album: str = "", playlist: str = "") -> str:
    """
    Play music on Spotify based on genre, artist, album, or playlist.

    Args:
        genre: The genre of music to play.
        artist: The artist to play.
        album: The album to play.
        playlist: The playlist to play.
    """
    return "Playing music on Spotify"