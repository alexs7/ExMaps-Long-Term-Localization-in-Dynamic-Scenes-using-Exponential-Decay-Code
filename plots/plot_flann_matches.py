import matplotlib.pyplot as plt
import numpy as np

#  Plot as in step 1 but groups in days / sessions
images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt"
all_sessions_dic = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/all_sessions_dic.npy')
all_sessions_dic = all_sessions_dic.item()

images_localised_labels = []
with open(images_localised_path) as f:
    images_localised_labels = f.readlines()
images_localised_labels = [x.strip() for x in images_localised_labels]

# load matching results (number of matches for each image)
results_all = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/results_all.npy').item()
results_base = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/results_base.npy').item()

results_all_over_sessions = []
results_base_over_sessions = []

for session_name, session_images in all_sessions_dic.items():
    temp_session_result_all = []
    temp_session_result_base = []
    for image_name, matches_no in results_all.items():
        if( image_name in session_images): # i.e is in current session
            temp_session_result_all.append(matches_no)
    for image_name, matches_no in results_base.items():
        if( image_name in session_images): # i.e is in current session
            temp_session_result_base.append(matches_no)


    results_all_over_sessions.append(np.sum(temp_session_result_all))
    results_base_over_sessions.append(np.sum(temp_session_result_base))

x = np.arange(len(all_sessions_dic.keys()))  # the label locations
labels = all_sessions_dic.keys()
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
# legends
rects1 = ax.bar(x - width/2, results_all_over_sessions, width, label='Against Complete Model')
rects2 = ax.bar(x + width/2, results_base_over_sessions, width, label='Against Base Model')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FLANN Matches - for each day')
ax.set_title('No. of feature matches of base and complete model (SFM images)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# fig.tight_layout()
plt.xticks(rotation=90)
fig.tight_layout()
plt.show()