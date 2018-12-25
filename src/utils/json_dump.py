import numpy
import json
import codecs


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_json(config, out_name, logger):
    """save json-file"""
    save_path = config['dataset']['output_directory'] + out_name + '.json'
    f = codecs.open(save_path, 'w', 'utf-8')
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
    logger.info(f'save json-file. {save_path}')
