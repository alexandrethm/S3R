import glob
import os

import pandas
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Band
from bokeh.models import ColumnDataSource, Select
from bokeh.models import TableColumn, DataTable
from bokeh.plotting import figure

grid_search_path = 'results/grid_search_1122_1128'

# output_file("test.html", title="Test")


# In[13]:


csv_files = sorted(glob.glob(os.path.join(grid_search_path, 'experiment*.csv')))
# Replace Nan values with '' to avoid JSON serialization error
csvs = [pandas.read_csv(c).fillna('None') for c in csv_files]

# Get csv file names in a dict
csv_names = {}
for i, f in enumerate(csv_files):
    csv_names[f[-10:-4]] = {
        'file_name': f,
        'id': i
    }


# In[14]:


def get_dataset(exp_id):
    df = csvs[exp_id]

    df['lower'] = df['valid_acc_mean'] - 2 * df['valid_acc_std']
    df['upper'] = df['valid_acc_mean'] + 2 * df['valid_acc_std']

    return ColumnDataSource(df.reset_index())


# In[15]:


def update_plot(attrname, old, new):
    exp = experiment_select.value

    exp_id = csv_names[exp]['id']
    src = get_dataset(exp_id)

    experiment_source.data.update(src.data)


# In[21]:


def make_plot(source):
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(plot_width=1000, plot_height=600, tools=TOOLS)
    p.scatter(x='0', y='valid_acc_mean', fill_alpha=0.3, size=5, source=source)

    band = Band(base='0', lower='lower', upper='upper', source=source, level='underlay',
                fill_alpha=1.0, line_width=1, line_color='black')
    p.add_layout(band)

    p.title.text = 'Grid Search {}'.format(grid_search_path)
    p.xgrid[0].grid_line_color = None
    p.ygrid[0].grid_line_alpha = 0.5
    p.xaxis.axis_label = 'epoch'
    p.yaxis.axis_label = 'valid_acc'

    return p


def get_summary_table():
    summary = pandas.DataFrame(columns=['exp_name', 'final_acc', 'final_std'])

    for i, c in enumerate(csvs):
        exp_name = [list(csv_names.keys())[i]]

        # with patience set to 100 in EarlyStopper callback, valid_acc is spposed not to move in the last 100 epochs
        c_final = c.iloc[-100:, :].mean()

        final_acc = [c_final['valid_acc_mean']]
        final_std = [c_final['valid_acc_std']]

        summary = pandas.concat([summary, pandas.DataFrame({
            'exp_name': exp_name,
            'final_acc': final_acc,
            'final_std': final_std,
        })])

    return ColumnDataSource(summary.reset_index())


# In[22]:


# Summary table
table_source = get_summary_table()
columns = [
    TableColumn(field="exp_name", title="Experiment"),
    TableColumn(field="final_acc", title="Accuracy"),
    TableColumn(field="final_std", title="Std"),
]
data_table = DataTable(source=table_source, columns=columns, width=600, height=600)

# Selected experiment graph
experiment_select = Select(value='0', title='Experiment', options=sorted(csv_names.keys()))

experiment_source = get_dataset(exp_id=0)
plot = make_plot(experiment_source)

experiment_select.on_change('value', update_plot)
controls = column(experiment_select)

# HTML layout
curdoc().add_root(row(column(plot, controls), data_table))
curdoc().title = "Grid Search Results"
