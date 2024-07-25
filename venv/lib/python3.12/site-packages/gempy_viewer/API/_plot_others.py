"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.

    Module with classes and methods to perform implicit regional modelling based on
    the potential field method.
    Tested on Ubuntu 16

    Created on 10/11/2019

    @author: Alex Schaaf, Elisa Heim, Miguel de la Varga
"""
import matplotlib.pyplot as plt

from gempy_viewer.optional_dependencies import require_pandas

try:
    import mplstereonet

    mplstereonet_import = True
except ImportError:
    mplstereonet_import = False



def plot_stereonet(self, litho=None, planes=True, poles=True,
                   single_plots=False,
                   show_density=False):
    if mplstereonet_import is False:
        raise ImportError(
            'mplstereonet package is not installed. No stereographic projection available.')

    from collections import OrderedDict

    if litho is None:
        litho = self.model._orientations.df['surface'].unique()

    if single_plots is False:
        fig, ax = mplstereonet.subplots(figsize=(5, 5))
        pn = require_pandas()
        df_sub2 = pn.DataFrame()
        for i in litho:
            df_sub2 = df_sub2.append(self.model._orientations.df[ self.model._orientations.df[ 'surface'] == i])

    for formation in litho:
        if single_plots:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='stereonet')
            ax.set_title(formation, y=1.1)

        # if series_only:
        # df_sub = self.model.orientations.df[self.model.orientations.df['series'] == formation]
        # else:
        df_sub = self.model._orientations.df[
            self.model._orientations.df['surface'] == formation]

        if poles:
            ax.pole(df_sub['azimuth'] - 90, df_sub['dip'], marker='o',
                    markersize=7,
                    markerfacecolor=self._color_lot[formation],
                    markeredgewidth=1.1, markeredgecolor='gray',
                    label=formation + ': ' + 'pole point')
        if planes:
            ax.plane(df_sub['azimuth'] - 90, df_sub['dip'],
                     color=self._color_lot[formation],
                     linewidth=1.5, label=formation + ': ' + 'azimuth/dip')
        if show_density:
            if single_plots:
                ax.density_contourf(df_sub['azimuth'] - 90, df_sub['dip'],
                                    measurement='poles', cmap='viridis',
                                    alpha=.5)
            else:
                ax.density_contourf(df_sub2['azimuth'] - 90, df_sub2['dip'],
                                    measurement='poles', cmap='viridis',
                                    alpha=.5)

        fig.subplots_adjust(top=0.8)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.9, 1.1))
        ax._grid(True, color='black', alpha=0.25)


