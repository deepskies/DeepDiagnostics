


'''
Want:
- random samples of sbi data plotted
- histogram of each measurement distribution
- 
'''

def plot_random_image(sim_dict):
    fontsize = 18
    if len(sim_dict['x'].shape) == 2:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex='col', sharey='row')
        for i in range(2):
            for j in range(2):
                batch_idx, set_idx = np.random.randint(0, len(y_global)), np.random.randint(0, len(y_local[0]))
                axs[i, j].imshow(x[batch_idx, set_idx])
                for i_param in range(y_global.shape[-1]):
                    glob_val = y_global[batch_idx, i_param] * y_global_std[i_param] + y_global_mean[i_param]
                    line = f"{config_dict['global_labels'][i_param]}: {glob_val.item():.2f}"
                    axs[i, j].annotate(line, xy=(2, 2 * (i_param + 2)), fontsize=fontsize, color='white')

                for i_param in range(y_local.shape[-1]):
                    loc_val = y_local[batch_idx, i_param] * y_local_std[i_param] + y_local_mean[i_param]
                    line = f"{config_dict['local_labels'][i_param]}: {loc_val[i_param].item():.2f}"
                    axs[i, j].annotate(line, xy=(2, 21 + 3 * (i_param + 2)), fontsize=fontsize, color='white')
    elif len(sim_dict['x'])
    fig.tight_layout()
    fig.savefig(fig_outdir + 'sample_images.png')
