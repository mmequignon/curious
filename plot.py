#!/home/pytorch/pytorch/sandbox/bin/python3


class Plot():
    def __init__(self, plot, data):
        self.plot = plot
        self.data = data
        for plot_name, vals in self.data["plots"].items():
            vals["line"] = vals["axes"].plot(
                vals["x_data"],
                vals["y_data"],
                vals.get("color", "k-"))[0]

    def update(self, new_values):
        for plot_name, vals in new_values.items():
            for val_name, val in vals.items():
                self.data["plots"][plot_name][val_name].append(val)
        for plot_name in self.data["plots"]:
            self.data["plots"][plot_name]["line"].set_xdata(
                self.data["plots"][plot_name]["x_data"])
            self.data["plots"][plot_name]["line"].set_ydata(
                self.data["plots"][plot_name]["y_data"])
        self.plot.draw()
        self.plot.pause(1e-6)

    @staticmethod
    def get_data_template(data_type):
        template = {
            "id": data_type[3],
            "x_label": data_type[0][1],
            "y_label": data_type[0][0],
            "x_data": [],
            "y_data": [],
            "x_limits": data_type[1],
            "y_limits": data_type[2],
            "title": data_type[4],
        }
        if data_type[0][0] == "accuracy":
            template["color"] = "g-"
        if data_type[0][0] == "loss":
            template["color"] = "r-"
        if data_type[0][0] == "errors":
            template["color"] = "b-"
        if data_type[0][0] == "fours":
            template["color"] = "k-"
        return template
