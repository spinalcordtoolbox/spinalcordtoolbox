

def import_existing_labels(self):
    def get_txt_files_in_output_directory(file_name, output_name):
        if output_name:
            (n, path) = self.seperate_file_name_and_path(self.file_name)
            output_file_name = path + output_name
        else:
            output_file_name = file_name
            output_file_name += '_ground_truth/'

        if os.path.exists(output_file_name):
            return (list(filter(lambda x: '.txt' in x, os.listdir(output_file_name))), output_file_name)
        else:
            return ([], output_file_name)

    def extract_coordinates(output_file_name, txt_file, file_name, output_name):
        if output_name:
            (n, path) = self.seperate_file_name_and_path(self.file_name)
            output_file_name = path + output_name + '/'
        else:
            output_file_name = file_name
            output_file_name += '_ground_truth/'
        file = open(output_file_name + txt_file, "r")
        list_coordinates = []
        for line in file:
            coordinates = ''
            for char in line:
                if char == ':':
                    list_coordinates.append(coordinates)
                    coordinates = ''
                else:
                    coordinates += char
            list_coordinates.append(coordinates)
        return list_coordinates

    def make_dic_labels():
        dic_labels = {'50': Coordinate([-1, -1, -1, 50]),
                      '49': Coordinate([-1, -1, -1, 49]),
                      '1': Coordinate([-1, -1, -1, 1]),
                      '3': Coordinate([-1, -1, -1, 3]),
                      '4': Coordinate([-1, -1, -1, 4]),
                      }
        for ii in range(5, 27):
            dic_labels[str(ii)] = Coordinate([-1, -1, -1, ii])
        return dic_labels

    def complete_dic_labels(dic_labels, list_coordinates):
        def update_max_label(current_label, max_label):
            if int(current_label) == 50:
                current_label = -1
            elif int(current_label) == 49:
                current_label = 0
            else:
                current_label = int(current_label)

            if current_label > max_label:
                max_label = current_label
            return max_label

        def remove_points_beyond_last_selected_label(dic_labels, max_label):
            for ikey in list(dic_labels.keys()):
                if ikey == '49':
                    if max_label == -1:
                        del dic_labels[ikey]
                else:
                    if max_label < int(ikey) and ikey != '50':
                        del dic_labels[ikey]
            return dic_labels

        def turn_string_coord_into_list_coord(coordinates):
            list_pos = []
            pos = ''
            for char in coordinates:
                if char == ',':
                    list_pos.append(pos)
                    pos = ''
                else:
                    pos += char
            list_pos.append(pos)
            return list_pos

        max_label = -5
        for coordinates in list_coordinates:
            list_pos = turn_string_coord_into_list_coord(coordinates)
            if list_pos[0] != '-1':
                max_label = update_max_label(list_pos[3], max_label)
            dic_labels[list_pos[3]] = Coordinate(
                [int(list_pos[0]), int(list_pos[1]), int(list_pos[2]), int(list_pos[3])])
            dic_labels = remove_points_beyond_last_selected_label(dic_labels, max_label)
        return dic_labels

    list_txt, path = get_txt_files_in_output_directory(self.file_name, self.output_name)
    for ilabels in list_txt:
        dic_labels = make_dic_labels()
        list_coordinates = extract_coordinates(path, ilabels, self.file_name, self.output_name)
        dic_labels = complete_dic_labels(dic_labels, list_coordinates)
        for ikey in list(dic_labels.keys()):
            self.main_pannel.main_plot.list_points.append(dic_labels[ikey])
    self.main_pannel.main_plot.draw_dots()
    self.main_pannel.second_plot.draw_lines()
    if self.main_pannel.main_plot.calc_list_points_on_slice():
        self.header.update_text('update', str(len(self.main_pannel.main_plot.calc_list_points_on_slice()) + 1))
        sct.printv('Output file you have chosen contained results of a previous labelling.\n'
                   'This data has been imported', type='info')



