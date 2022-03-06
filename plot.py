import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import numpy as np
import params
from matplotlib.backends.backend_pdf import PdfPages
font = {'size'   : 24}
matplotlib.rcParams['figure.figsize'] = (16, 12)
p = params.Parameters(SELECTED_MODEL = 'VLSTM', SELECTED_DATASET = 'HIGHD', UNBALANCED = False, ABLATION = False)
prediction_seq = p.SEQ_LEN - p.IN_SEQ_LEN + 1
matplotlib.rc('font', **font)

# CL figure
with PdfPages('cl.pdf') as export_pdf:
    fig = plt.figure()
    ax_all = fig.add_subplot(1, 1, 1)

    ax_all.scatter(np.arange(p.CL_EPOCH+1),((p.END_SEQ_CL - p.START_SEQ_CL)/p.FPS)[:p.CL_EPOCH+1] , label = 'Max included TTLC',  marker ='o', linewidth = 10)
    ax_all.scatter(np.arange(p.CL_EPOCH+1), p.LOSS_RATIO_CL[:p.CL_EPOCH+1], label = 'MTL Loss ratio ($\gamma$)',  marker = '*' , linewidth = 10)

    # And a corresponding grid
    ax_all.grid(True)
    #ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS
    #plt.xlim(ttlc_seq[0], ttlc_seq[-1])
    #plt.ylim(0,100)
    plt.xlabel('Epoch Number')
    plt.ylabel('CL parameter')
    plt.tight_layout()
    ax_all.legend(loc = 'lower right')
    export_pdf.savefig()
    
# ROC curve
with PdfPages('roc_all.pdf') as export_pdf:
    folder_dir = './results/figures/roc'
    file_names = [
        'HIGHD_REGIONATTCNN3ks3cn16mcTruet2clTruecsTruecvFalse.pickle',
        'HIGHD_MLPhd512toFalset0clFalsecsFalsecvFalsewirth.pickle',
        'HIGHD_MLPhd512toFalset0clFalsecsFalsecvFalseshou.pickle',
        'HIGHD_VLSTMln1toFalsehd512t0clFalsecsFalsecvFalsewirth.pickle', 
        'HIGHD_VLSTMln1toFalsehd512t0clFalsecsFalsecvFalseours.pickle', 
        'CSLSTM.pickle']
    names = ['Proposed','MLP1', 'MLP2','LSTM1','LSTM2','CSLSTM']
    axs = []
    for i, file_name in enumerate(file_names):
        full_dir = os.path.join(folder_dir, file_name)
        with open(full_dir,'rb') as fid:
            axs.append(pickle.load(fid))
    plt.close('all')    
    fig = plt.figure()
    ax_all = fig.add_subplot(1, 1, 1)
    i =0
    for ax in axs:
        line = ax.axes[0].lines[0]
        ax_all.plot(line.get_data()[0], line.get_data()[1], label = names[i],  linewidth=5)
        
        #plt.title(names[i])
        #plt.grid()
        i+=1
    major_ticks = np.arange(0, 101, 20)/100
    minor_ticks = np.arange(0, 101, 5)/100

    ax_all.set_xticks(major_ticks)
    ax_all.set_xticks(minor_ticks, minor=True)
    ax_all.set_yticks(major_ticks)
    ax_all.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax_all.grid(which='both')

    # Or if you want different settings for the grids:
    ax_all.grid(which='minor', alpha=0.2)
    ax_all.grid(which='major', alpha=0.5)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.tight_layout()
    ax_all.legend(loc = 'lower right')
    export_pdf.savefig()
    



# recall curve
with PdfPages('recall_all.pdf') as export_pdf:
    folder_dir = './results/figures/recall_vs_TTLC'
    file_names = [
        'HIGHD_REGIONATTCNN3ks3cn16mcTruet2clTruecsTruecvFalse.pickle',
        'HIGHD_MLPhd512toFalset0clFalsecsFalsecvFalsewirth.pickle',
        'HIGHD_MLPhd512toFalset0clFalsecsFalsecvFalseshou.pickle',
        'HIGHD_VLSTMln1toFalsehd512t0clFalsecsFalsecvFalsewirth.pickle', 
        'HIGHD_VLSTMln1toFalsehd512t0clFalsecsFalsecvFalseours.pickle', 
        'CSLSTM.pickle']
    names = ['Proposed','MLP1', 'MLP2','LSTM1','LSTM2','CSLSTM']
    axs = []
    for i, file_name in enumerate(file_names):
        full_dir = os.path.join(folder_dir, file_name)
        with open(full_dir,'rb') as fid:
            axs.append(pickle.load(fid))
    plt.close('all')    
    fig = plt.figure()
    ax_all = fig.add_subplot(1, 1, 1)
    i =0
    for ax in axs:
        line = ax.axes[0].lines[0]
        ax_all.plot(line.get_data()[0], line.get_data()[1], label = names[i],  linewidth=5)
        
        #plt.title(names[i])
        #plt.grid()
        i+=1

    # And a corresponding grid
    ax_all.grid(True)
    ttlc_seq = (prediction_seq-np.arange(prediction_seq))/p.FPS
    plt.xlim(ttlc_seq[0], ttlc_seq[-1])
    plt.ylim(0,100)
    plt.xlabel('Time to lane change (TTLC) (s)')
    plt.ylabel('Recall (%)')
    ax_all.legend(loc = 'lower right')
    plt.tight_layout()
    export_pdf.savefig()
