from matplotlib import pyplot as plt
import numpy as np

setting = 'informer_custom_ftMS_sl96_ll48_pl2_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_cspTrue_dilateTrue_passthroughTrue_Exp_0'
# exp = Exp_Informer(args)
# exp.predict(setting, True)
 
# preds = np.load('results/' + setting + '/pred.npy')
trues = np.load('results/' + setting + '/real_prediction.npy')
batchy = np.load('results/' + setting + '/batchy.npy')

print(trues)
print(trues.shape)
print(batchy)
print(batchy.shape)
# print(preds)
 
# print(trues.shape)
# print(preds.shape)
 
# plt.figure()
# plt.plot(trues[0,:,-1].reshape(-1),label='GroundTruth')
# plt.plot(preds[0,:,-1].reshape(-1),label='Prediction')
# plt.legend()
# plt.savefig('results/' + setting + '/prediction.png')
