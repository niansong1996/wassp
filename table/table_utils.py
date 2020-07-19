from texttable import Texttable
from table.utils import load_jsonl

table_file = "../data/wikisql/processed_input/preprocess_4/tables.jsonl"




def display_table(table):
    kg = table['kg']
    column_names = ['row_idx'] + kg['row_0'].keys()
    all_row_names = kg.keys()

    rows = []
    for row_name in all_row_names:
        row = kg[row_name]
        row_val_list = [row_name]
        for column_name in column_names[1:]:
            assert len(row[column_name]) == 1
            row_val_list.append(row[column_name][0])
        rows.append(row_val_list)

    rows = sorted(rows, key=lambda x: int(x[0][4:]))
    table_info = [column_names] + rows

    t = Texttable()
    t.set_max_width(300)
    t.add_rows(table_info)
    print t.draw()




if __name__ == '__main__':
    tables = load_jsonl(table_file)
    table_dict = dict([(table['name'], table) for table in tables])

    while True:
        user_input = raw_input("Input the table name:")
        table_name = user_input
        #table_name = user_input[2:-1]
        display_table(table_dict[table_name])
