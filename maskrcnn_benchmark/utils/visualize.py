from visdom import Visdom
import numpy as np
import logging

class Visualizer(object):

    def __init__(self, loss_keys, env, port, hostname):
        logger = logging.getLogger("maskrcnn_benchmark.visualize")
        logger.info('Launching visdom server ...')

        self.viz = Visdom(env=env, port=port, server=hostname)
        assert self.viz.check_connection(timeout_seconds=3), \
            'No connection could be formed quickly'

        self.title = env
        self.loss_keys = loss_keys
        self.colors, self.lines = self._get_loss_line_attribute(len(loss_keys))
        self.loss_log_dict = {}

    def _get_loss_line_attribute(self, cnt):
        COLORS = [[244,  67,  54],
                [233,  30,  99],
                [156,  39, 17],
                [103,  58, 183],
                [ 63,  81, 181],
                [ 33, 150, 243],
                [  3, 169, 244],
                [  0, 188, 212],
                [  0, 150, 136],
                [ 76, 175,  80],
                [139, 195,  74],
                [205, 220,  57],
                [255, 235,  59],
                [255, 193,   7],
                [255, 152,   0],
                [255,  87,  34]]
        LINES = ['solid', 'dash', 'dashdot']

        assert cnt < len(COLORS)

        colors = [COLORS[i] for i in range(0, cnt)]
        lines = [LINES[i%3] for i in range(0, cnt)]

        return np.array(colors), np.array(lines)

    def _moving_average(self, data, window=40):
        if len(data) < 2*window:
            return data
        ret = np.cumsum(data, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        return ret[window-1:] / window

    def add_data_point(self, loss_dict_reduced):
        X = []
        Y = []
        for key in self.loss_keys:
            if key not in self.loss_log_dict:
                self.loss_log_dict[key] = []
            self.loss_log_dict[key].append(loss_dict_reduced[key].item())
            y = self._moving_average(self.loss_log_dict[key])
            x = np.arange(0, len(y))
            X.append(x)
            Y.append(y)

        self.viz.line(
            X=np.array(X).transpose(1,0),
            Y=np.array(Y).transpose(1,0),
            opts={
                'dash': self.lines,
                'linecolor': self.colors,
                'legend': self.loss_keys,
                'title': self.title + ' training losses'
            },
            win = 1,
        )
    
