class VoyageError(Exception):
    @property
    def user_message(self) -> str: ...

class APIError(VoyageError): ...
class TryAgain(VoyageError): ...
class Timeout(VoyageError): ...
class APIConnectionError(VoyageError): ...
class InvalidRequestError(VoyageError): ...
class MalformedRequestError(VoyageError): ...
class AuthenticationError(VoyageError): ...
class RateLimitError(VoyageError): ...
class ServerError(VoyageError): ...
class ServiceUnavailableError(VoyageError): ...
