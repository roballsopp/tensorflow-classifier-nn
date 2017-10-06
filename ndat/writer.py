import logging
import os
from ndat import Header

class NdatWriter:
	def __init__(self, example_builder):
		self._example_builder = example_builder

	def to_file(self, output_path_name):
		output_dir = os.path.dirname(output_path_name)
		os.makedirs(output_dir, exist_ok=True)
		output_filepath = output_path_name + '.ndat'
		num_examples_to_be_written = self._example_builder.num_examples
		feature_width = self._example_builder.feature_width
		label_width = self._example_builder.label_width

		logging.info('Writing ' + str(num_examples_to_be_written) + ' training examples of feature length ' + str(feature_width) + ', and label length ' + str(label_width) + ' to ' + str(output_filepath))

		header = Header(
			feature_width=feature_width,
			label_width=label_width,
			num_examples=num_examples_to_be_written,
			feature_format=Header.FMT_FLOAT,
			label_format=Header.FMT_FLOAT,
			label_offset=self._example_builder.label_offset
		)

		with open(output_filepath, 'wb') as file:
			header_bytes = header.to_bytes()

			file.write(header_bytes)

			self._example_builder.reset()

			while self._example_builder.has_next():
				features, labels = self._example_builder.get_next_example()

				features.tofile(file)
				labels.tofile(file)

		logging.info('Write completed successfully!')
