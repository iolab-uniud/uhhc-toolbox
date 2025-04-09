import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from ..models import Instance, Solution
import re
import bisect

plt.style.use('ggplot')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "CMS",
    'font.size' : 14,
    "text.latex.preamble": r"\usepackage{fontawesome5}"
})

def plot(instance: Instance, solution: Solution, patient_height : float = 0.3):
    figsize = (12, patient_height * (len(instance.patients) + len(instance.terminal_points)))

    def select_hatch(p, served_patient):
        starts = [tw.start for tw in p.time_windows] 
        idx = bisect.bisect_right(starts, served_patient.start_service_time) - 1 
        start = p.time_windows[idx].start
        end = p.time_windows[idx].end
        if served_patient.service == "lunch_break":
            return r'\\\\' 
        if served_patient.start_service_time >= start and served_patient.start_service_time <= end:
            return None
        if served_patient.start_service_time < start:
            return r'\\\\'
        else:
            return r'////' 
    
    def select_hatch_color(p, served_patient):
        starts = [tw.start for tw in p.time_windows] 
        idx = bisect.bisect_right(starts, served_patient.start_service_time) - 1 
        start = p.time_windows[idx].start
        end = p.time_windows[idx].end
        if served_patient.service == "lunch_break":
             return 'white'
        if served_patient.start_service_time >= start and served_patient.start_service_time <= end:
            return 'gray'    
        else:
            return 'white'
        
    def latexify_label(label):
        m = re.match('(\w)(\d+)', label)
        if label == "lunch_break":
            return "LB"
        if not m:
            return label
        return f"${m.group(1)}_{{{m.group(2)}}}$"

    _, ax = plt.subplots(1, figsize=figsize)
    
    palette = cm.Dark2(np.linspace(0, 1, len(instance.caregivers))) #cm.viridis(np.linspace(0, 1, len(instance.caregivers)))
    np.random.shuffle(palette)
    c_dict = {f"{c.id}": palette[i] for i, c in enumerate(instance.caregivers)}

    # setup departing points
    for i, _ in enumerate(instance.terminal_points):
        ax.barh([f"$_{{{i}}}$" + r'\faWarehouse'], [0], left=[0], height=0.1, color="darkgray", align='center', zorder=1)
    for p in instance.patients:
        for tw in p.time_windows:
            ax.barh([latexify_label(p.id)], [tw.end - tw.start], left=[tw.start], height=0.45 if len(p.required_services) > 1 else 0.45, color="darkgray", align='center', zorder=1)

    for r in solution.routes:
        for visit in r._visits:
            patient = instance._patients[visit.patient]
            height = 0.4 if len(patient.required_services) == 1 else 0.4 if visit.service == patient.required_services[0].service else -0.4
            align = 'center' if len(patient.required_services) == 1 else 'edge'
            ax.barh([latexify_label(visit.patient)],
                    [visit.end_service_time - visit.start_service_time], 
                    left=[visit.start_service_time], 
                    color=c_dict[r.caregiver_id], 
                    height=height, 
                    align=align, 
                    hatch=[select_hatch(patient, visit)], 
                    edgecolor=[select_hatch_color(patient, visit)])        

    for r in solution.routes:        
        c = r.caregiver_id
        prev = (r.locations[0].departing_time, instance._terminal_points[instance._caregivers[c].departing_point].distance_matrix_index)
        for visit in r._visits:
            patient = instance._patients[visit.patient]
            if len(patient.required_services) == 1:
                offset = 0
            elif visit.service == patient.required_services[0].service:
                offset = 0.2
            else:
                offset = -0.2            
            cur = (visit.arrival_at_patient, patient.distance_matrix_index + offset)
            ax.annotate('', xytext=cur, xy=prev, arrowprops=dict(arrowstyle="<-", color=c_dict[c], lw=2, alpha=0.6), zorder=0)             
            ax.text(cur[0], cur[1], f'{latexify_label(visit.service)}', fontsize=7, bbox=dict(boxstyle="round", fc="w", alpha=0.6), ha='left', va='center')
            ax.barh([latexify_label(visit.patient)], [visit.start_service_time - cur[0]], left=cur[0], height=offset * 2, align='center' if len(patient.required_services) == 1 else 'edge', color='white', hatch='ooo', alpha=0.3, edgecolor=c_dict[c], zorder=2)
            prev = (visit.end_service_time, patient.distance_matrix_index + offset)
        cur = (r.locations[-1].arrival_time, instance._terminal_points[instance._caregivers[c].departing_point].distance_matrix_index)
        ax.annotate('', xytext=cur, xy=prev, arrowprops=dict(arrowstyle="<-", color=c_dict[c], lw=2, alpha=0.6))        

    plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0)
    plt.legend(handles=[mpatches.Patch(color=c_dict[c.id], label=latexify_label(c.id)) for c in instance.caregivers], loc='upper left', bbox_to_anchor=(1, 1))
    return plt