import time
import logging


class RetryRunner:
	def __init__(self, max_retries=5, retry_interval=1.0):
		self._retry_count = 0
		self._max_retries = max_retries
		self._retry_interval = retry_interval

	def run(self, func, error=Exception):
		try:
			return func()
		except error as e:
			if self._retry_count < self._max_retries:
				logging.warning('Caught error, retrying...', e)
				self._retry_count += 1
				time.sleep(self._retry_interval)
				return self.run(func, error)
			else:
				raise
