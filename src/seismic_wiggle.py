import numpy as np
import matplotlib.pyplot as plt
from .helpers import insert_zeros_in_trace, input_check, input_check_color_dicts, input_check_picks_color, \
    input_check_picks_markers, input_check_marker_curve
from functools import wraps
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Slider


def set_plt_params(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_font_size = plt.rcParams['font.size']
        old_serif = plt.rcParams['font.sans-serif']

        plt.rcParams['font.size'] = kwargs.get('font_size', 20)
        plt.rcParams['font.sans-serif'] = 'Arial'
        try:
            func(*args, **kwargs)
        finally:
            plt.rcParams['font.size'] = old_font_size
            plt.rcParams['font.sans-serif'] = old_serif
        return

    return wrapper


class PGather:

    def _init_gui(
            self,
            ax,
            fig_width,
            fig_height,
    ):

        if isinstance(ax, type(None)):
            _, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='w')

        self.fig = ax.get_figure()
        self.ax = ax

        # s_gain_axs = plt.axes()
        self.s_gain_axs = self.fig.add_axes([0.13, 0.02, 0.75, 0.03])
        self.s_gain = Slider(self.s_gain_axs, 'gain', 0, 10, valinit=1, valfmt='%d')
        self.s_gain.on_changed(lambda x: self.update(gain=x))

        return ax

    @set_plt_params
    def __init__(
            self,
            traces,
            dt=1.,
            picks=None,
            offset=None,
            mask=None,
            gain=1.,
            clip=None,
            max_norm=True,
            start_time=0,
            time_label='time',
            traces_label='trace',
            title='',
            time_vertical=False,
            fill_positive=True,
            fill_negative=False,
            trace_color=None,
            invert_y_axis=False,
            picks_legend=True,
            picks_on_amplitude=False,
            picks_marker=True,
            picks_marker_size=5,
            picks_marker_color=None,
            picks_marker_fill=True,
            picks_curve=False,
            picks_style_dash=False,
            picks_dash_style='dashed',
            picks_curve_style_dash=False,
            picks_curve_line_style='solid',
            picks_curve_marker=True,
            offset_tick_rotation=0,
            alpha=.5,
            mask_alpha=.5,
            mask_cmap=None,
            mask_vmin=0,
            mask_vmax=1.1,
            dist_for_3c=.5,
            font_size=20,
            ax=None,
            fig_width=10,
            fig_height=10,
            offset_ticks_freq=1,
            display=False,
    ):
        self.plt_area_pos = {}
        self.plt_area_neg = {}
        self.plt_line = {}

        self.clip = clip
        self.gain = gain
        self.dt = dt

        self.start_time = start_time

        traces, offset, mask, picks = input_check(traces, offset, mask, picks)
        offset_ticks = offset.copy()
        offset = np.arange(traces.shape[0])

        self.traces_ip = traces
        self.offset = offset
        self.mask = traces
        self.picks = picks

        trace_color, fill_positive, fill_negative = input_check_color_dicts(
            traces.shape[2],
            **dict(
                trace_color=trace_color,
                fill_positive=fill_positive,
                fill_negative=fill_negative,
            )
        )

        self.trace_color = trace_color
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative

        ax = self._init_gui(ax, fig_width, fig_height)

        # TODO special dict for picks properties
        picks_marker_color, picks_dash_style, picks_curve_line_style = input_check_picks_color(
            picks, picks_marker_color,
            picks_dash_style,
            picks_curve_line_style
        )
        if picks_marker_fill:
            picks_marker_fill = picks_marker_color
        else:
            picks_marker_fill = {x: 'None' for x in picks_marker_color}

        picks_marker = input_check_picks_markers(picks, picks_marker)
        picks_curve_marker = input_check_picks_markers(picks, picks_curve_marker)
        picks_curve = input_check_marker_curve(picks, picks_curve)

        if display:
            print('picks_marker', picks_marker)
            print('picks_marker_fill', picks_marker_fill)
            print('picks_curve_marker', picks_curve_marker)
            print('picks_curve', picks_curve)

        if max_norm:
            traces /= np.nanmax(np.abs(traces) * 2)

        traces *= np.float32(np.abs(gain))

        if clip:
            traces[traces > clip] = clip
            traces[traces < -clip] = -clip

        k = np.float32(traces.shape[2] * dist_for_3c) + 1e-15
        traces_lim = np.array([-.5, traces.shape[0] - .5], dtype=np.float32)
        traces_lim += np.array([- np.nanmax(np.abs(traces[0])), np.nanmax(np.abs(traces[-1]))]) / k

        time_lim = np.array([0, traces.shape[1]]) * dt + start_time

        dist_for_3c = (traces.shape[2] > 1) * dist_for_3c + (traces.shape[2] == 1) * 1.

        if invert_y_axis:
            fill_positive, fill_negative = fill_negative, fill_positive
            dist_for_3c *= -1.

        self.dist_for_3c = dist_for_3c

        if time_vertical:
            def _get_x_y(_x): return _x[::-1]

            _fill = ax.fill_betweenx
            _set_time_label = ax.set_ylabel
            _set_time_lim = ax.set_ylim
            _set_traces_lim = ax.set_xlim
            _set_traces_label = ax.set_xlabel
            _set_trace_ticks = ax.set_xticks
            _set_trace_tick_labels = ax.set_xticklabels
            _set_grid_axis = 'x'

        else:
            def _get_x_y(_x):
                return _x

            _fill = ax.fill_between
            _set_time_label = ax.set_xlabel
            _set_time_lim = ax.set_xlim
            _set_traces_lim = ax.set_ylim
            _set_traces_label = ax.set_ylabel
            _set_trace_ticks = ax.set_yticks
            _set_trace_tick_labels = ax.set_yticklabels
            _set_grid_axis = 'y'

        self._get_x_y = _get_x_y

        time = np.arange(traces.shape[1]) * dt

        for jt in range(traces.shape[0]):
            off = offset[jt]

            for jc in range(traces.shape[2]):
                trace = traces[jt, :, jc]
                if any(fill_positive.values()) | any(fill_negative.values()):
                    trace, time = insert_zeros_in_trace(trace)
                    time = time * dt

                trace /= (traces.shape[2] * dist_for_3c)
                shift = .5 + (jc - traces.shape[2]) / (traces.shape[2] + 1)
                shift *= dist_for_3c
                shift += off

                if any(fill_positive.values()):
                    vertices = [list(zip(time + self.start_time, trace * (trace >= 0) + shift))]
                    self.plt_area_pos[jt] = PolyCollection(
                        vertices,
                        alpha=alpha,
                        facecolor=fill_positive[jc],
                    )
                    ax.add_collection(self.plt_area_pos[jt])

                if any(fill_negative.values()):
                    vertices = [list(zip(time + self.start_time, trace * (trace <= 0) + shift))]
                    self.plt_area_neg[jt] = PolyCollection(
                        vertices,
                        alpha=alpha,
                        facecolor=fill_negative[jc],
                    )
                    ax.add_collection(self.plt_area_pos[jt])

                x, y = _get_x_y([time + start_time, trace + shift])
                self.plt_line[jt] = ax.plot(x, y, color=trace_color[jc])[0]

                for ip, label in enumerate(picks):
                    p = picks[label]
                    pick_time = p[jt] * dt + start_time

                    if picks_style_dash | (traces.shape[2] > 1):
                        pick_amplitude = [off - .3, off + .3]
                        pick_time = [pick_time, pick_time]
                        marker = None
                    else:
                        pick_amplitude = off + trace[np.int32(p[jt])] * picks_on_amplitude
                        marker = picks_marker[label]

                    x, y = _get_x_y([pick_time, pick_amplitude])
                    ax.plot(
                        x,
                        y,
                        color=picks_marker_color[label],
                        markeredgecolor=picks_marker_color[label],
                        markerfacecolor=picks_marker_fill[label],
                        marker=marker,
                        markersize=picks_marker_size,
                        linestyle=picks_dash_style,
                    )

        if picks:
            j_traces = np.arange(traces.shape[0])
            for ip, label in enumerate(picks):
                if picks_curve[label]:
                    p = np.int32(picks[label])
                    pick_times = p * dt + start_time

                    if picks_curve_style_dash:
                        pick_amplitudes = np.insert(offset - .5, np.arange(len(offset)) + 1, offset + .5)
                        pick_times = np.insert(pick_times, np.arange(len(offset)) + 1, pick_times)
                        marker = None

                    else:
                        pick_amplitudes = offset + traces[j_traces, p, 0] * picks_on_amplitude
                        marker = picks_curve_marker[label]

                    x, y = _get_x_y([pick_times, pick_amplitudes])
                    ax.plot(
                        x,
                        y,
                        color=picks_marker_color[label],
                        label=label,
                        markeredgecolor=picks_marker_color[label],
                        markerfacecolor=picks_marker_fill[label],
                        marker=marker,
                        linestyle=picks_curve_line_style,
                    )
                else:
                    ax.plot(
                        np.nan,
                        np.nan,
                        color=picks_marker_color[label],
                        label=label,
                        markeredgecolor=picks_marker_color[label],
                        markerfacecolor=picks_marker_fill[label],
                        marker=picks_marker[label],
                        linestyle='None'
                    )

        if not isinstance(mask, type(None)):
            if isinstance(mask_cmap, type(None)):
                mask_cmap = plt.cm.Greys
            # c_map.set_bad('green', 1.)
            extent = (time_lim[0], time_lim[1], -.5, traces.shape[0] - .5)

            mask /= np.nanmax(np.abs(mask))
            if time_vertical:
                extent = ( -.5, traces.shape[0] - .5, time_lim[0], time_lim[1])
                mask = mask.T

            kwargs = dict(
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap=mask_cmap,
                alpha=mask_alpha,
                vmin=mask_vmin,
                vmax=mask_vmax,
            )

            ax.imshow(mask, **kwargs)

        _set_time_label(time_label, fontsize=font_size)
        _set_time_lim(time_lim)
        _set_traces_label(traces_label, fontsize=font_size)
        _set_traces_lim(traces_lim)
        _set_trace_ticks(offset[::offset_ticks_freq])
        _set_trace_ticks(np.arange(traces.shape[0] - 1) + .5, minor=True)
        _set_trace_tick_labels(offset_ticks[::offset_ticks_freq], rotation=offset_tick_rotation)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.grid(which='minor', axis=_set_grid_axis)
        ax.set_title(title)

        # TODO: handle with timestamps

        # ax.set_xticks(np.arange(0, data.shape[0], 5))
        # if np.any(np.array(xticklabel)):
        #     ticks = np.array(ax.get_xticks(), dtype=int)
        #     ticks = ticks[ticks < len(xticklabel)]
        #     xticklabel = np.array(xticklabel)[ticks]
        #     ax.set_xticklabels(xticklabel)

        if invert_y_axis:
            ax.invert_yaxis()

        if (len(picks) > 0) & picks_legend:
            ax.legend(loc=2)

        ax.get_figure().show()
        self.ax = ax

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if len(self.plt_line) == 0:
            return

        traces = self.traces_ip.copy()
        traces *= np.float32(np.abs(self.gain))

        if self.clip:
            traces[traces > self.clip] = self.clip
            traces[traces < -self.clip] = -self.clip

        time = np.arange(traces.shape[1]) * self.dt

        for jt in range(traces.shape[0]):
            off = self.offset[jt]

            for jc in range(traces.shape[2]):
                trace = traces[jt, :, jc]

                if any(self.fill_positive.values()) | any(self.fill_negative.values()):
                    trace, time = insert_zeros_in_trace(trace)
                    time = time * self.dt

                trace /= (traces.shape[2] * self.dist_for_3c)
                shift = .5 + (jc - traces.shape[2]) / (traces.shape[2] + 1)
                shift *= self.dist_for_3c
                shift += off

                if any(self.fill_positive.values()):
                    vertices = [list(zip(time + self.start_time, trace * (trace >= 0) + shift))]
                    self.plt_area_pos[jt].set_verts(vertices)

                if any(self.fill_negative.values()):
                    vertices = [list(zip(time + self.start_time, trace * (trace <= 0) + shift))]
                    self.plt_area_neg[jt].set_verts(vertices)

                _, y = self._get_x_y([time + self.start_time, trace + shift])
                self.plt_line[jt].set_ydata(y)

        self.ax.get_figure().canvas.draw_idle()


