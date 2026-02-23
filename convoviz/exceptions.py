"""Custom exceptions for convoviz."""


class ConvovizError(Exception):
    """Base exception for all convoviz errors."""


class InvalidZipError(ConvovizError):
    """Raised when a ZIP file is invalid or missing conversation data."""

    def __init__(self, path: str, reason: str = "missing conversation data") -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Invalid ZIP file '{path}': {reason}")


class ConfigurationError(ConvovizError):
    """Raised for configuration-related errors."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        super().__init__(message)


class MessageContentError(ConvovizError):
    """Raised when message content cannot be extracted."""

    def __init__(self, message_id: str) -> None:
        self.message_id = message_id
        super().__init__(f"No valid content found in message: {message_id}")
