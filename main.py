from Sub_Functions.Evaluate import open_popup
from Sub_Functions.Analysis import Analysis
from Sub_Functions.Read_data import Read_data
from Sub_Functions.Plot import ALL_GRAPH_PLOT

DB = ["DB1"]

Choose = open_popup("Do you need Complete Execution:")

if Choose == "Yes":
    for i in range(len(DB)):
        # Read_data(DB[i])

        TP = Analysis(DB[i])

        # TP.COMP_Analysis()

        TP.PERF_Analysis()

        Plot = ALL_GRAPH_PLOT()

        Plot.GRAPH_RESULT(DB[i])

else:

    for i in range(len(DB)):
        Plot = ALL_GRAPH_PLOT()

        Plot.GRAPH_RESULT(DB[i])
