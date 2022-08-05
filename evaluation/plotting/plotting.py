from random import randint
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import operator


class PLOTTER(): 
    
    def __init__(self):
        
        ### define colors ##
        self.blue2    = "#2e518c"
        self.blue3    = "#5079b3"
        self.blue4    = "#7da7d9"
        self.green1   = "#146614"
        self.green2   = "#2e8c2e"
        self.green3   = "#50b350"
        self.green4   = "#7dd97d"
        self.green5   = "#b3ffb3"
        self.red1     = "#660000"
        self.red2     = "#8c1919"
        self.red3     = "#b33e3e"
        self.red4     = "#d97272"
        self.red5     = "#ffb3b3"
        self.magenta1 = "#581466"
        self.magenta2 = "#762e8c"
        self.magenta3 = "#9650b3"
        self.magenta4 = "#b87dd9"
        self.magenta5 = "#dfb3ff"
        self.orange1  = "#b34e0b"
        self.orange2  = "#c67322"
        self.orange3  = "#d99a3d"
        self.orange4  = "#ecc05c"
        self.orange5  = "#ffe480"
        self.cyan1    = "#146666"
        self.cyan2    = "#2e8c8c"
        self.cyan3    = "#50b3b3"
        self.cyan4    = "#7dd9d9"
        self.cyan5    = "#b3ffff"
        self.gray1    = "#4d4d4d"
        self.gray2    = "#6c6c6c"
        self.gray3    = "#8c8c8c"
        self.gray4    = "#acacac"
        self.gray5    = "#cccccc"
        
        palette = "viridis"
        sns.set_palette(palette)
        
        # Plot customizations
        self.tic_font_size = "x-small"
        self.label_font_size = "small"

        self.rc = {
            "figure.figsize":(10.75, 3.75),
            "font.family":"sans serif", 
            "text.usetex":False,
            "xtick.labelsize":self.tic_font_size,
            "ytick.labelsize":self.tic_font_size,
            "axes.axisbelow":True,
            "lines.linewidth":0.8, 
            "legend.fancybox":True, 
            # "text.usetex" : True, 
            # "pdf.fonttype" : 42
        }

        matplotlib.rcParams["lines.linewidth"] = 0.8
        matplotlib.rcParams["legend.fancybox"] = True
        # matplotlib.rcParams["pdf.fonttype"] = 42
        # matplotlib.rcParams["ps.fonttype"] = 42

        # matplotlib.rcParams["text.usetex"] = True

        sns.set(rc=self.rc)
        sns.set_style('white')

        plt.tight_layout()


    def plot_lfps(
        self,
        data,                   # pd dataframe to take data from
        csc_nrs,                # channel IDs for LFP data
        start,                  # start time in ms
        stop,                   # stop time in ms
        x_name="movie_times",   # df column name for labels for x axis
        y_name="filtered"       # df column name of LFP data
    ):

        data = data.copy()

        len_c = len(csc_nrs)
        lst_c = list(range(len_c))

        #spike_colors = ["red", "blue", "green", "orange", "violet"]
        lfp_colors = []
        for i in range(len_c):
            lfp_colors.append('#%06X' % randint(0, 0xFFFFFF))

        ## set up frame for data ##

        fig, axs = plt.subplots(len_c,1, figsize=(16,len_c*2))

        for i in lst_c: 
            ## data ##
            cs = csc_nrs[i]
            movie_times = data[(data['csc_nr']== cs)][x_name].to_numpy()[0][start:stop]#S/1000
            lfp_dt = data[(data['csc_nr']== cs)][y_name].to_numpy()[0][start:stop]
            #df_lfp_i = df_lfps[(df_lfps['csc_nr']== cs) & (df_lfps['timestamps'] > lfp_time_start) & (df_lfps['timestamps'] < lfp_time_end)]

            ## fig prep ##
            axs[i].set_title(f'csc_nr : {cs}', fontweight='bold')
            axs[i].set_xlabel('movie times [s]', fontweight='light')
            axs[i].set_ylabel('voltage', fontweight='light')

            ## plot ##
            axs[i].plot(movie_times, lfp_dt, color=lfp_colors[i])

        plt.subplots_adjust(hspace=0.75)

        return fig



    def plot_spikes(
        self,
        data,                   # pd dataframe (df) to take data from
        unit_idx,               # indices for spike data
        start,                  # start time in ms
        stop,                   # stop time in ms
        x_times,                # labels for x axis
        y_name                  # column name of spike data
    ):

        data = data.copy()

        len_u = len(unit_idx)
        lst_u = list(range(len_u))
        spike_linelengths=[0.5] * len_u

        #spike_colors = ["red", "blue", "green", "orange", "violet"]
        spike_colors = []
        for i in range(len_u):
            spike_colors.append('#%06X' % randint(0, 0xFFFFFF))


        ## set up frame for data ##

        fig, axs = plt.subplots(1,1, figsize=(16,len_u*0.75))
        axs.set_yticks(lst_u)
        axs.set_yticklabels(data["unit_id"][unit_idx])
        axs.set_ylabel('units', fontweight='light')
        axs.set_xlabel('recording time', fontweight='light')
        axs.set_title('spikes selected by index', fontweight='bold')


        ## get data ##

        plt_spike_dt_time = []
        min_time = 1000000000000000
        max_time = 0
        for i in unit_idx: 
            if y_name=="bin1_cnt":
                c_spike_times = x_times[i][start:stop][
                    (data[y_name][i][start:stop]).astype(dtype=bool)
                ]
            plt_spike_dt_time.append(c_spike_times)
            if len(c_spike_times) != 0:
                c_min = min(plt_spike_dt_time[-1])
                c_max = max(plt_spike_dt_time[-1])
                min_time = c_min if (c_min<min_time) else min_time
                max_time = c_max if (c_max>max_time) else max_time


        ## plot data ##

        axs.eventplot(plt_spike_dt_time, color=spike_colors, linelengths=spike_linelengths)

        for i in lst_u: 
            axs.plot([min_time, max_time], [i, i], "black", linewidth=0.2)

        return fig



    def plot_lfp_with_spikes(
        self,
        lfps,            # pd dataframe (df) to take lfp data from
        spikes,          # pd dataframe (df) to take spike data from
        start,           # start time in ms
        stop,            # stop time in ms
        x_name,          # labels for x axis taken from lfp df
        y_lfp,           # df column name of LFP data
        y_spike          # column name of spike data
    ): 

        lfps = lfps.copy()
        spikes = spikes.copy()

        x_times = lfps[x_name]
        unit_idx = lfps.index[lfps["csc_nr"].isin(csc_nrs)].to_numpy()

        len_c = len(csc_nrs)
        lst_c = list(range(len_c))

        #len_u = len(unit_idx)
        #lst_u = list(range(len_u))
        spike_linelength = 1

        #spike_colors = ["red", "blue", "green", "orange", "violet"]
        lfp_colors = []
        for i in range(len_c):
            lfp_colors.append('#%06X' % randint(0, 0xFFFFFF))

        ## set up frame for data ##

        fig, axs = plt.subplots(len_c,1, figsize=(4,len_c/2+0.5))
        if len_c>1:
            #fig.suptitle("LFP data and spikes for different channels", fontweight='bold')#, fontsize=20)
            fig.supxlabel('movie times [s]', fontweight='light')#, fontsize=20)
            fig.supylabel('voltage', fontweight='light')#, fontsize=20)
        else:
            #axs.set_title("LFP data and spikes for different channels", fontweight='bold')#, fontsize=20)
            axs.set_xlabel('movie times [s]', fontweight='light', fontsize=self.label_font_size)
            axs.set_ylabel('voltage', fontweight='light', fontsize=self.label_font_size)

        plt_spike_dt_time = []
        min_time = 1000000000000000
        max_time = 0


        for i, j in zip(lst_c, unit_idx): 
            ## data ##
            cs = csc_nrs[i]
            movie_times = lfps[(lfps['csc_nr']== cs)][x_name].to_numpy()[0][start:stop]/1000
            lfp_dt = lfps[(lfps['csc_nr']== cs)][y_lfp].to_numpy()[0][start:stop]

            c_spike_times = x_times[j][start:stop][
                (spikes[y_spike][j][start:stop]).astype(dtype=bool)
            ]/1000      

            ## fig prep ##
            if len_c>1: 
                axi = axs[i] 
                axi.set_title(f'channel {cs}', fontweight='bold')

            else:
                axi = axs

            axi.grid(linewidth=0.5, linestyle="dashed", zorder=0)
            axi.tick_params(
                    direction = "in", 
                    bottom = False, top = False,
                    left = True, right = False,
                    zorder = 1
            )

            #axs[i].set_xlabel('movie times [s]', fontweight='light')
            #axs[i].set_ylabel('voltage', fontweight='light')

            ## plot ##
            axi.plot(movie_times, lfp_dt, color=lfp_colors[i])
            axi.set_ylim(int(lfp_dt.min())-800, int(lfp_dt.max())+300)
            axsj = axi.twinx()

            axsj.set_ylim(-1, 2)
            axsj.set_yticks([-0.5])
            axsj.set_yticklabels(["spikes"], fontsize=self.tic_font_size)
            axsj.eventplot(c_spike_times, lineoffset=-1, linelengths=spike_linelength)#, color=lfp_colors[i], linelengths=spike_linelengths)


        plt.subplots_adjust(hspace=0.70)

        return fig
    
    
    def plot_LPF_spike_data(
        self,
        lfps,           
        spike_times,
        units,
        spike_rates,
        #sta,
        start,           # start time in ms
        stop,            # stop time in ms
        movie_times
    ): 

        spike_linelength = 0.5
        n_spike_times = len(spike_times)

        #spike_colors = ["red", "blue", "green", "orange", "violet"]
        spike_colors = [self.magenta2, self.red2, self.orange2, self.green2, self.blue2]
        lfp_colors = [self.gray2]

        ## set up frame for data ##

        fig, axs = plt.subplots(2,1, figsize=(7.5,3))
        
        ##### LFP data #####

        #axs.set_title("LFP data and spikes for different channels", fontweight='bold')#, fontsize=20)
        #axs[0].set_xlabel('movie times [s]', fontweight='light', fontsize=self.label_font_size)
        axs[0].set_ylabel('voltage', fontweight='light', fontsize=self.label_font_size)

        plt_spike_dt_time = []
        min_time = 1000000000000000
        max_time = 0
        
        ## data ##
        lfp_dt = lfps[start:stop]
        c_spike_times = []
        for i in range(n_spike_times):
            c_spike_times.append(movie_times[start:stop][
                (spike_times[i][start:stop]).astype(dtype=bool)
            ]/1000)
        movie_times_dt = movie_times[start:stop]/1000

        ## fig prep ##
        axs[0].grid(linewidth=0.5, linestyle="dashed", zorder=0)
        axs[0].tick_params(
                direction = "in", 
                bottom = False, top = False,
                left = True, right = False,
                zorder = 1
        )

        #axs[i].set_xlabel('movie times [s]', fontweight='light')
        #axs[i].set_ylabel('voltage', fontweight='light')

        ## plot ##
        axs[0].plot(movie_times_dt, lfp_dt, color=lfp_colors[0], label="LFP trace")
        axs[0].set_ylim(int(lfp_dt.min())-150, int(lfp_dt.max())+150)
        #axsj = axs[0].twinx()

        #axsj.set_ylim(-0.75, 2)
        #axsj.set_yticks([-0.5])
        #axsj.set_yticklabels(["spikes"], fontsize=self.tic_font_size)
        #axsj.eventplot(c_spike_times, lineoffset=-0.5, linelengths=spike_linelength, color=spike_colors[0])#, color=lfp_colors[i], linelengths=spike_linelengths)
        axs[0].legend(fontsize=self.tic_font_size, loc="lower right")
        
        ##### spike rate data #####
        
        #axs.set_title("LFP data and spikes for different channels", fontweight='bold')#, fontsize=20)
        axs[1].set_xlabel('movie times [$s$]', fontweight='light', fontsize=self.label_font_size)
        axs[1].set_ylabel('spike rate\n[spikes per $r$ ms]', fontweight='light', fontsize=self.label_font_size)

        plt_spike_dt_time = []
        min_time = 1000000000000000
        max_time = 0
        
        ## data ##
        spike_rates_dt = []
        for (sr, r) in spike_rates: 
            spike_rates_dt.append((sr[int(start/r):int(stop/r)], r))        

        ## fig prep ##
        axs[1].grid(linewidth=0.5, linestyle="dashed", zorder=0, axis="x")
        axs[1].tick_params(
                direction = "in", 
                bottom = False, top = False,
                left = True, right = False,
                zorder = 1
        )

        #axs[i].set_xlabel('movie times [s]', fontweight='light')
        #axs[i].set_ylabel('voltage', fontweight='light')

        ## plot ##
        for i in range(len(spike_rates_dt)):
            (srdt, r) = spike_rates_dt[i]
            movie_times_dt = movie_times[start:stop]/1000
            axs[1].plot(movie_times_dt, np.repeat(srdt, r), color=spike_colors[i+1], label=f"spike rate ($r$={r})")
        axs[1].set_ylim(-(spike_linelength*n_spike_times*3.5)-(n_spike_times), int(srdt.max())*2)
        axsj = axs[1].twinx()

        axsj.set_ylim(-(spike_linelength*1.5)-(n_spike_times*(spike_linelength*1.5)), 4)
        tiks = []
        tik_labels = []
        for i in range(n_spike_times):
            u = units[i]
            tiks.append(-(spike_linelength*1.5)*(i+1))
            tik_labels.append(f"unit {u}")
            axsj.eventplot(c_spike_times[i], lineoffset=-(spike_linelength*1.5)*(i+1), linelengths=spike_linelength, color=spike_colors[0])#, color=lfp_colors[i], linelengths=spike_linelengths)
        
        axsj.set_yticks(tiks)
        axsj.set_yticklabels(tik_labels, fontsize=self.tic_font_size)

        axs[1].legend(
            bbox_to_anchor=(1.0, 1.0), 
            fontsize=self.tic_font_size, 
            loc="upper right", 
            ncol=len(spike_rates_dt)
        )
        

        #plt.subplots_adjust(hspace=0.70)

        return fig
    
    
    def plot_cv_loss(
        self, 
        df
    ):
        
        ## set up frame for data ##
        fig, ax = plt.subplots(1,1, figsize=(4,2))
        ax.grid(linewidth=0.5, linestyle="dashed", zorder=0, axis="x")
        
        ## fig prep ##
        ax.tick_params(
                direction = "in", 
                bottom = False, top = False,
                left = True, right = False,
                zorder = 1
        )

        ax.set_xlabel('', fontweight='light', fontsize=self.label_font_size)
        ax.set_ylabel('', fontweight='light', fontsize=self.label_font_size)
        
        sns.lineplot(y="losses", x="alpha", hue="unit", err_style="bars", data=df, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, labels=labels, 
            title="units", title_fontsize=self.label_font_size,
            fontsize=self.tic_font_size, 
            loc="upper right", bbox_to_anchor=(1.25,1)
        )                      
        
        return fig
    
    
    def plot_reg_loss(
        self, 
        df
    ):
        
        ## set up frame for data ##
        fig, ax = plt.subplots(1,1, figsize=(7,2))
        ax.grid(linewidth=0.5, linestyle="dashed", zorder=0, axis="x")
        
        ## fig prep ##
        ax.tick_params(
                direction = "in", 
                bottom = False, top = False,
                left = True, right = False,
                zorder = 1
        )

        ax.set_xlabel('', fontweight='light', fontsize=self.label_font_size)
        ax.set_ylabel('', fontweight='light', fontsize=self.label_font_size)
        
        sns.boxplot(y="losses [NLL]", x="alpha", hue="regularization", data=df, ax=ax)
        #sns.lineplot(y="losses [NLL]", x="alpha", hue="unit", style="regularization", err_style="bars", data=df, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, labels=labels, 
            title="units", title_fontsize=self.label_font_size,
            fontsize=self.tic_font_size, 
            loc="upper right", bbox_to_anchor=(2.1,1), 
            ncol = 3
        )                      
        
        return fig


    def plot_reg_losses_bp(
        self, 
        df
    ):
        
        ## set up frame for data ##
        fig, ax = plt.subplots(1,1, figsize=(4,2))
        ax.grid(linewidth=0.5, linestyle="dashed", zorder=0, axis="y")
        
        ## fig prep ##
        ax.tick_params(
                direction = "in", 
                bottom = False, top = False,
                left = True, right = False,
                zorder = 1
        )

        ax.set_xlabel('', fontweight='light', fontsize=self.label_font_size)
        ax.set_ylabel('', fontweight='light', fontsize=self.label_font_size)
        
        palette = {"L1":self.orange3,
           "L2":self.cyan3, 
           "None":"tab:purple"}
              
        sns.boxplot(y="validation costs\nper unit [NLL]", x="alpha", hue="regularization", data=df, ax=ax, linewidth=0.5, fliersize=2, palette=palette)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, labels=labels, 
            title="regularization", title_fontsize=self.label_font_size,
            fontsize=self.tic_font_size, 
            loc="upper right", bbox_to_anchor=(1,1), 
            ncol = 3
        )                      
        
        return fig, ax
    
    
    def plot_cv_costs(
        self, 
        df
    ):
        
        ## set up frame for data ##
        fig, ax = plt.subplots(1,1, figsize=(4,2))
        ax.grid(linewidth=0.5, linestyle="dashed", zorder=0)
        
        ## fig prep ##
        ax.tick_params(
                direction = "in", 
                bottom = False, top = False,
                left = True, right = False,
                zorder = 1
        )

        ax.set_xlabel('', fontweight='light', fontsize=self.label_font_size)
        ax.set_ylabel('', fontweight='light', fontsize=self.label_font_size)
        
        palette = {"L1":self.orange3,
           "L2":self.cyan3, 
           "None":"tab:purple"}
        
        sns.lineplot(y="mean validation\ncosts [NLL]", x="alpha", hue="regularization", data=df, ax=ax, palette=palette)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, labels=labels, 
            title="regularization", title_fontsize=self.label_font_size,
            fontsize=self.tic_font_size, 
            loc="upper right", bbox_to_anchor=(1,1), 
            ncol = 3
        )                      
        
        return fig, ax
    
    
    
    def plot_reg_losses_line(
        self, 
        df
    ):
        
        ## set up frame for data ##
        fig, ax = plt.subplots(1,1, figsize=(7,2))
        ax.grid(linewidth=0.5, linestyle="dashed", zorder=0, axis="x")
        
        ## fig prep ##
        ax.tick_params(
                direction = "in", 
                bottom = False, top = False,
                left = True, right = False,
                zorder = 1
        )

        ax.set_xlabel('', fontweight='light', fontsize=self.label_font_size)
        ax.set_ylabel('', fontweight='light', fontsize=self.label_font_size)
        
        sns.lineplot(y="losses [NLL]", x="alpha", hue="unit", style="regularization", err_style="bars", data=df, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, labels=labels, 
            title="units", title_fontsize=self.label_font_size,
            fontsize=self.tic_font_size, 
            loc="upper right", bbox_to_anchor=(2.1,1), 
            ncol = 3
        )                      
        
        return fig
    
    
    
    def plot_theta(
        self, 
        df, 
        alpha
    ):
        
        ## set up frame for data ##
        fig, ax = plt.subplots(1,1, figsize=(7,2))
        ax.grid(linewidth=0.5, linestyle="dashed", zorder=0)
        
        ## fig prep ##
        ax.tick_params(
                direction = "in", 
                bottom = False, top = False,
                left = True, right = False,
                zorder = 1
        )

        ax.set_xlabel('', fontweight='light', fontsize=self.label_font_size)
        ax.set_ylabel('', fontweight='light', fontsize=self.label_font_size)
        ax.set_title(f"alpha {alpha}")
        sns.lineplot(y="weight values", x="weight indices", hue="unit", style="regularization", err_style="bars", data=df, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles, labels=labels, 
            title="units", title_fontsize=self.label_font_size,
            fontsize=self.tic_font_size, 
            loc="upper right", bbox_to_anchor=(2.1,1), 
            ncol = 2
        )                      
        
        return fig
    
    

    def plot_STA(
        self, 
        lfps, 
        spikes,
        interval        
    ):
        
        ## set up frame for data ##
        fig, axs = plt.subplots(5, 4, figsize=(12, 7))
        
        x_label = "time around spike [$ms$]"
        y_label = "STA"
        
        #fig.suptitle("STAs for different channels with 100ms before and after", fontweight='bold', fontsize=30)
        #fig.supxlabel('STA interval', fontweight='light', fontsize=20)
        #fig.supylabel('mean voltage', fontweight='light', fontsize=20)
        #fig, axs = plt.subplots(4, 2, figsize=(20, 10))
        #for i in range(len(df_lfps[f"STA_int{intervall}"])): 
        for i in range(20): 
                
            k = i // 4
            j = i % 4
        
            ## fig prep ##
            axs[k,j].grid(linewidth=0.5, linestyle="dashed", zorder=0)
            axs[k,j].tick_params(
                    direction = "in", 
                    bottom = False, top = False,
                    left = True, right = False,
                    zorder = 1
            )
            
            axs[k,j].set_xlabel('', fontweight='light', fontsize=self.label_font_size)
            axs[k,j].set_ylabel('', fontweight='light', fontsize=self.label_font_size)

            c = list(set(lfps.csc_nr.tolist()))[i]

            c_idx = np.where(lfps.csc_nr == c)[0]
        
            df_STA = pd.DataFrame({
                y_label : np.array(lfps[y_label][c_idx].values.tolist()).flatten(), 
                "units" : np.repeat(spikes.unit_id.to_numpy()[c_idx], interval*2+1), 
                x_label : np.array(
                    (list(
                        map(
                            operator.neg, 
                            reversed(range(1,interval+1))
                        )
                    ) 
                     + list(range(interval+1)))

                    * c_idx.size) 
            })
            colors = sns.color_palette([self.red1, self.red2, self.red3, self.red4])
            #sns.set_palette(sns.color_palette(colors))
            sns.lineplot(ax=axs[k,j], data=df_STA, x=x_label, y=y_label, hue="units", 
                        palette="dark:salmon_r")

            if c_idx.size>1:
                df_STA_mean = pd.DataFrame({
                    y_label : df_lfps[y_label][c_idx].to_numpy().mean(axis=0), 
                    "units" : np.repeat(np.array("mean"), interval*2+1), 
                    x_label : np.array(
                        (list
                         (map(
                             operator.neg, 
                             reversed(range(1,interval+1))
                         )) 
                         + list(range(interval+1)))) #len(df_spikes.unit_i)
                })
                
                colors = [self.gray4]
                sns.set_palette(sns.color_palette(colors))
                sns.lineplot(ax=axs[k,j], data=df_STA_mean, x=x_label, y=y_label, hue="units")

            axs[k,j].set_title(f"channel {c}", fontsize=self.label_font_size)
            handles, labels = axs[k,j].get_legend_handles_labels()
            axs[k,j].legend(
                handles=handles, labels=labels, 
                title="units", title_fontsize=self.label_font_size,
                fontsize=self.tic_font_size, 
                loc="upper left", bbox_to_anchor=(1.02,1.09)
            ) 

        plt.subplots_adjust(hspace=0.9, wspace=0.95)
        
        return fig
    
    
    

    def plot_best_theta(
        self, 
        theta_df, 
        channels,
        best_values       
    ):
        
        ## set up frame for data ##
        rows = 5
        cols = 4
        fig, axs = plt.subplots(rows, cols, figsize=(12, 7))
        
        x_label = "time before spike [$ms$]"
        y_label = "weight values"

        csc_set = list(set(channels))
        n_csc = len(csc_set)

        for i in range(n_csc): 

            k = i // 4
            j = i % 4

            ## fig prep ##
            axs[k,j].grid(linewidth=0.5, linestyle="dashed", zorder=0)
            axs[k,j].tick_params(
                    direction = "in", 
                    bottom = False, top = False,
                    left = True, right = False,
                    zorder = 1
            )

            axs[k,j].set_xlabel('', fontweight='light', fontsize=self.label_font_size)
            axs[k,j].set_ylabel('', fontweight='light', fontsize=self.label_font_size)

            c = csc_set[i]
        
            x_c = theta_df.iloc[np.array(theta_df.channel==c)]
            xs_u = []
            for u in set(x_c.unit.tolist()):
                l, alpha = best_values[f"{u}"]
                x_u = x_c.iloc[np.array(x_c.unit==u)]
                x_u = x_u.iloc[np.array(x_u.alpha==alpha)]
                x_u = x_u.iloc[np.array(x_u.regularization==f"L{l+1}")]
                xs_u.append(x_u)
            xs = pd.concat(xs_u)
            sns.lineplot(ax=axs[k,j], data=xs, y=y_label, x=x_label, hue="unit", palette="dark:salmon_r")


            axs[k,j].set_title(f"channel {c}", fontsize=self.label_font_size)
            handles, labels = axs[k,j].get_legend_handles_labels()
            axs[k,j].legend(
                handles=handles, labels=labels, 
                title="units", title_fontsize=self.label_font_size,
                fontsize=self.tic_font_size, 
                loc="upper left", bbox_to_anchor=(1.02,1.09)
            ) 
            
        plt.subplots_adjust(hspace=0.9, wspace=0.95)
        
        return fig, axs
    
