import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

def plot_tokens_sec(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Tokens Per Second')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    single_mean, single_std = None, None
    device0_mean, device0_std =  None, None
    device1_mean, device1_std =  None, None
    # plot([28.414332, 27.808804, 49.08619475364685],
    #     [0.410696, 0.214668, 0.007076],
    #     ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
    #     'ddp_vs_rn.png')

    # plot_tokens_sec([161793.82, 82514.61707],
    #     [984.23, 478.23],
    #     ['Data Parallel - Two GPU', 'Single GPU'],
    #     'ddp_vs_rn_token.png')

    pp_mean, pp_std = 14.184119701385498, 0.14570045471191406
    mp_mean, mp_std = 15.521642684936523, 0.142592191696167
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png')
    
    plot_tokens_sec([45125.64407536964, 41236.22932683023],
        [463.5343609168158, 378.8235],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_mp_token_sec.png')