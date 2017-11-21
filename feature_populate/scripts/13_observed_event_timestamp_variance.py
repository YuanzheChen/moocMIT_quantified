"""
Created on July 02, 2013

@author: Colin for ALFA, MIT lab: colin2328@gmail.com
Feature 13- Variance of a students observed event timestamps in one week

Modifications:
 2013-07-04 - Franck Dernoncourt - franck.dernoncourt@gmail.com - fixed a few typos + add a TODO which needs to be fixed
"""
# make this as high as possible until MySQL quits on you
BLOCK_SIZE = 1000


def main(** kwargs):
    """
    """
    # Parse the kwargs
    conn = kwargs['conn']
    conn2 = kwargs['conn2']
    db_name = kwargs['db_name']
    start_date = kwargs['start_date']
    current_date = kwargs['current_date']
    import_path = kwargs['import_dir']
    num_weeks = kwargs['num_weeks']
    parent_conn = kwargs['parent_conn']
    # Dynamic import the required module
    try:
        import math
        import sys
        sys.path.append(import_path)
        from processor.processor import block_sql_command
    except RuntimeWarning:
        pass

    cursor2 = conn2.cursor()

    first_row = '''SELECT observed_events.user_id,
             FLOOR((UNIX_TIMESTAMP(observed_events.observed_event_timestamp) -
             UNIX_TIMESTAMP('%s')) / (3600 * 24 * 7))
             AS week, observed_event_timestamp
             FROM `%s`.observed_events AS observed_events
             INNER JOIN `%s`.user_dropout as u
             ON u.user_id = observed_events.user_id
             WHERE
             u.dropout_week IS NOT NULL
             AND
             observed_events.validity = 1
             AND FLOOR((UNIX_TIMESTAMP(observed_events.observed_event_timestamp)
                - UNIX_TIMESTAMP('%s')) / (3600 * 24 * 7)) < '%s'
             GROUP BY observed_events.user_id, week, observed_event_timestamp
             ASC LIMIT 1
          ''' % (start_date, db_name, db_name, start_date, num_weeks)

    cursor2.execute(first_row)
    first = cursor2.fetchone()
    cursor2.close()
    times = []
    old_week = first[1]
    old_user_id = first[0]

    cursor = conn.cursor()

    sql = '''SELECT COUNT(*)
            FROM `%s`.observed_events AS observed_events''' % db_name
    cursor.execute(sql)
    n = int(cursor.fetchone()[0])
    cursor.close()

    cursor = conn.cursor()

    # get all the observed events times for a user for a week

    sql = '''SELECT observed_events.user_id,
             FLOOR((UNIX_TIMESTAMP(observed_events.observed_event_timestamp) -
             UNIX_TIMESTAMP('%s')) / (3600 * 24 * 7))
             AS week, observed_event_timestamp
             FROM `%s`.observed_events AS observed_events
             INNER JOIN `%s`.user_dropout as u
             ON u.user_id = observed_events.user_id
             WHERE
             u.dropout_week IS NOT NULL
             AND
             observed_events.validity = 1
             AND FLOOR((UNIX_TIMESTAMP(observed_events.observed_event_timestamp)
                - UNIX_TIMESTAMP('%s')) / (3600 * 24 * 7)) < '%s'
             GROUP BY observed_events.user_id, week, observed_event_timestamp
             ASC
          ''' % (start_date, db_name, db_name, start_date, num_weeks)

    cursor.execute(sql)

    data_to_insert = []

    used_index = 0

    for i in xrange(n):
        row = cursor.fetchone()
        if not row:
            break
        user_id = row[0]
        week = row[1]
        timestamp = row[2]

        if week != old_week or user_id != old_user_id:
            # if either have changed, compute entropy for list,
            # insert entropy into database, and clear list before adding
            entropy = compute_deviation(times)
            data_to_insert.append((user_id, week, entropy, current_date))
            times = []
            used_index += 1

        time = timestamp.time()
        seconds = ((time.hour * 60 + time.minute) * 60) + time.second
        times.append(seconds)
        old_week = week
        old_user_id = user_id

    cursor.close()

    sql = "INSERT INTO `%s`.user_long_feature(feature_id," % db_name
    sql = sql + '''
        user_id,
        feature_week,
        feature_value,
        date_of_extraction)
        VALUES (13, %s, %s, %s, %s)
        '''
    cursor = conn.cursor()
    block_sql_command(conn, cursor, sql, data_to_insert, BLOCK_SIZE)
    cursor.close()
    conn.commit()

    if parent_conn:
        parent_conn.send(True)


def compute_deviation(times):
    import math
    mean = sum(times, 0.0) / len(times)
    d = [(i - mean) ** 2 for i in times]
    std_dev = math.sqrt(sum(d) / len(d))
    return std_dev
