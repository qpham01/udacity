from matplotlib import pyplot as plt

training_losses = [135.68341, 90.179314, 88.047424, 85.326019, 83.943146, 80.450867, 77.443802, 74.403633, \
                   71.202652, 66.987518, 62.953705, 58.564613, 52.98365, 48.442932, 43.487549, 38.901138, \
                   33.214096, 29.233784, 24.111528, 20.383381]
validation_accuracies = [0.2913797501093634, 0.1539148177169653, 0.075618464686879092, 0.065416985463842212, \
                       0.071792909981446801, 0.07638357562796555, 0.074980872235973703, 0.092323386910100946, \
                       0.136189747574195, 0.13542463658370402, 0.1689619995431309, 0.16959959200163696, \
                       0.18923743945911994, 0.23833205819403569, 0.24011731703376868, 0.31841367010375854, \
                       0.36330017864749736, 0.39517980105595391, 0.46608008183985455, 0.55763835775946446]
loss_plot = plt.subplot(211)
loss_plot.set_title('Training Losses')
loss_plot.plot(training_losses, 'g', label='Training Losses')
loss_plot.set_xlim([0, len(training_losses)])
loss_plot.legend(loc=1)
acc_plot = plt.subplot(212)
acc_plot.set_title('Validation Accuracies')
acc_plot.plot(validation_accuracies, 'x' , label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([0, len(validation_accuracies)])
acc_plot.legend(loc=2)
plt.tight_layout()
plt.show()
