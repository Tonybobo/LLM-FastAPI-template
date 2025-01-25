import logging
import sys

class Logger:
    _instance = None
    _logger = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._logger = None
            cls._instance._initialize_logger() 
        return cls._instance

    def _initialize_logger(self) -> None:
        """Initialize logger with console handlers"""
        if self._logger is None:
            self._logger = logging.getLogger("ArticleSummarizer")
            self._logger.setLevel(logging.DEBUG)

            if self._logger.handlers:
                self._logger.handlers.clear()

            console_formater = logging.Formatter(
                '%(levelname)s: %(message)s'
            )

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formater)

            self._logger.addHandler(console_handler)

    def get_logger(self):
        """Get back Logger instance instead of create new class"""
        if self._logger is None:
            self._intialize_logger()
        return self._logger
    
    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
    

def get_logger():
    return Logger.instance().get_logger()
        
