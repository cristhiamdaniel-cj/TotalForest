# src/logger.py
import logging

class CustomLogger:
    def __init__(self, name='src'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add handlers to the logger
        if not self.logger.handlers:
            self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger
