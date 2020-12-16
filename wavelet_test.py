import pywt
import matplotlib.pyplot as plt

# herd average daily data
resting_other = [28.91339036524438, 29.743780277696214, 29.548863555664244, 28.198968111619028, 23.555879427468266, 16.152823216427205, 11.855278325797151, 13.975960193547476, 18.772238009139247, 18.712357974041424, 20.960513821730213, 21.137531628001128, 18.08575578135764, 20.040312715374775, 20.903118962306536, 16.718056726041063, 11.232885023168569, 11.913087615089177, 24.086679574116133, 35.187223671934156, 33.45383755582282, 32.06214168093001, 29.89823217974353, 29.07008183567694]
resting_hot = [28.563464433677566, 27.920349620921446, 28.56012485175542, 25.46486350246531, 19.72637707120466, 13.865905884100302, 11.223364921337541, 12.120756041740275, 11.359731933289375, 11.174529161706428, 13.58052015563712, 14.489100225714331, 14.058820749515604, 15.326934718374675, 15.934396568602079, 12.428173109164275, 6.791350392805945, 9.800822293284604, 27.69379589393452, 32.60797493963323, 30.941512308519762, 31.251200021603765, 28.504904729649137, 28.325431456431282]
eating_other = [1.4919467734927627, 0.8523586571593762, 0.4766381664761891, 0.6243339255322851, 0.7712789001248006, 1.4670838612262158, 6.300207367542366, 4.890276683755925, 3.357038733163084, 3.237476178568794, 3.010981580600291, 2.901734124289815, 4.112849942277719, 3.3673131174179196, 3.109000089336111, 5.117733333760203, 7.124416680534803, 7.17948687287643, 4.218713111707272, 1.6943811515081626, 1.6437187166598002, 1.8252519256895121, 1.825492087335791, 1.8525396500990046]
eating_hot = [1.5649496924839155, 1.1244106785607393, 0.7655203919282936, 1.2390141482153265, 1.06709319098853, 1.9302873525711854, 5.4931419299813, 4.7849097705275145, 4.88891839403016, 6.232827820247678, 5.284710610823937, 5.095319859485516, 5.520335668490235, 4.92885835356811, 4.3676420616292395, 6.300638886325942, 8.459849628797818, 8.09644505555093, 4.256272405466652, 1.8789041490479717, 2.247183071650235, 1.8181975709267342, 2.137170401942538, 2.1638690541826917]
panting_other = [0.8305604449324365, 0.5318436267415947, 0.3542587996376067, 0.8960814955808648, 2.532859702462344, 3.6497127224395634, 3.611990038165759, 5.285439778606979, 6.477865785497869, 6.765575692425326, 7.073887001415041, 7.47314054359492, 6.900231653499905, 6.729141574713174, 6.483842324627946, 5.122674972938758, 3.3553754455010503, 2.196423347900797, 1.405952906113506, 0.8059778894349816, 0.8804479479899624, 0.917131649995086, 0.9720724776886432, 0.8969175874525778]
panting_hot = [0.512715840735248, 0.4647082254082774, 0.30595431253895994, 1.3694761312158643, 3.8346276838739155, 5.105714420737813, 5.070302250167092, 7.607419092776917, 7.636194856953824, 8.592399975695765, 12.74443646805456, 13.283922973578145, 12.316834958491517, 11.903149873865521, 11.533286675203156, 7.704910580668681, 4.921830378943532, 2.9287413331773053, 1.1415181595392996, 0.992517446164994, 0.9768232114445943, 0.93423679076079, 0.8921702106592072, 0.7067739053530077]

# animal single day data
resting_single = [46, 60, 29, 32, 36, 22, 8, 3, 23, 27, 40, 8, 8, 5, 7, 1, 1, 1, 1, 7, 42, 44, 5, 6]
eating_single = [0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 1, 20, 3, 7, 2, 8, 11, 6, 0, 0, 12, 8]
panting_single = [0, 0, 0, 5, 0, 6, 5, 5, 3, 11, 20, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1]

# select state:
state_other = eating_other
state_hot = eating_hot
state_single = eating_single

coefs_other = pywt.wavedec(state_other, 'db1', level=2)
cA2o, cD2o, cD1o = coefs_other

coefs_hot = pywt.wavedec(state_hot, 'db1', level=2)
cA2h, cD2h, cD1h = coefs_hot

coefs_comp = pywt.wavedec(state_single, 'db1', level=2)
cA2c, cD2c, cD1c = coefs_comp

plt.figure(2, figsize=(8,7))
plt.subplot(4, 1, 1)
plt.plot(state_other, label = 'other')
plt.plot(state_hot, label = 'hot')
plt.plot(state_single, label = 'comp')
plt.title("Signal")
plt.legend(loc="upper right")

plt.subplot(4,1,2)
plt.plot(cA2o, label = 'other')
plt.plot(cA2h, label = 'hot')
plt.plot(cA2c, label = 'comp')
plt.title("Second Approximate")

plt.subplot(4,1,3)
plt.plot(cD2o, label = 'other')
plt.plot(cD2h, label = 'hot')
plt.plot(cD2c, label = 'comp')
plt.title("Second Detail")

plt.subplot(4,1,4)
plt.plot(cD1o, label = 'other')
plt.plot(cD1h, label = 'hot')
plt.plot(cD1c, label = 'comp')
plt.title("First Detail")

plt.tight_layout()
plt.show()