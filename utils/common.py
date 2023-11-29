import configparser


def get_exercise_config(exercise, exercise_config):
    config_parser = configparser.RawConfigParser()
    config_parser.read(exercise_config)
    important_parts = config_parser.get(exercise, "important_parts").split(",")
    peak_parts = config_parser.get(exercise, "peak_parts").split(",")
    valid_classes = config_parser.get(exercise, "valid_classes").split(",")
    return important_parts, peak_parts, valid_classes
