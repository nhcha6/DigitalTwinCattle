
def area_under_graph(signal):
    # store area under graph between 5am and 8pm for each day
    areas = []
    for i in range(int(len(signal)/24)):
       daily_signal = signal[i*24:i*24+24]
       area = sum(daily_signal[6:20])
       areas.append(area)
    print(areas)
    return areas

def heat_stress_comp(signal, herd, animal_ave):
    signal_areas = area_under_graph(signal)
    herd_areas = area_under_graph(herd)
    animal_area = area_under_graph(animal_ave)[0]

    herd_comp = [b / m for b,m in zip(signal_areas, herd_areas)]
    animal_ave_comp = [x/animal_area for x in signal_areas]

    return herd_comp, animal_ave_comp