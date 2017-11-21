'''
Created on Nov 21, 2013
@author: Colin Taylor colin2328@gmail.com
Feature 202- A student's average number of attempts as compared with other students as a percentile
Requires that populate_feature_9_average_number_of_attempts.sql has already been run!
'''

BLOCK_SIZE = 1000


def main(**kwargs):
    """
    """
    # Parse the kwargs
    conn = kwargs['conn']
    db_name = kwargs['db_name']
    current_date = kwargs['current_date']
    import_path = kwargs['import_dir']
    parent_conn = kwargs['parent_conn']
    # Dynamic import the required module
    try:
        from scipy.stats import percentileofscore
        import sys
        sys.path.append(import_path)
        from processor.processor import block_sql_command
    except RuntimeWarning:
        pass
    # numWeeks doesn't do anything here, but python scripts are automatically
    # called so we need the arg
    cursor = conn.cursor()

    sql = '''SELECT user_id, feature_week,feature_value
          FROM `%s`.user_long_feature
          WHERE feature_id = 9
          AND date_of_extraction >= '%s'
          ''' % (db_name, current_date)

    cursor.execute(sql)

    week_values = {}
    data = []
    for [user_id, week, value] in cursor:
        data.append((user_id, week, value))
        if week in week_values:
            week_values[week].append(value)
        else:
            week_values[week] = [value]

    data_to_insert = []
    for i, [user_id, week, value] in enumerate(data):
        data_to_insert.append((user_id, week,
            percentileofscore(week_values[week], value), current_date))
    cursor.close()

    sql = "INSERT INTO `%s`.user_long_feature(feature_id, user_id," % db_name

    sql = sql + '''
                feature_week,
                feature_value,
                date_of_extraction)
                VALUES (202, %s, %s, %s, %s)
                '''

    cursor = conn.cursor()
    block_sql_command(conn, cursor, sql, data_to_insert, BLOCK_SIZE)
    cursor.close()
    conn.commit()

    if parent_conn:
        parent_conn.send(True)
    return True
