from feature_dict import get_feature_ids, get_feature_dict
from processor.processor import open_sql_connection, run_sql_script, \
    close_sql_connection, run_python_script
from processor.processor import color_message


def populate(cfg_mysql, cfg_pipeline, cfg_course, cfg_mysql_script_path):
    feature_list = cfg_pipeline['exec']
    for feature_id in feature_list:
        populate_single_feature(feature_id, cfg_mysql, cfg_pipeline,
                                cfg_course, cfg_mysql_script_path)


def populate_single_feature(feature_id, cfg_mysql, cfg_pipeline,
                            cfg_course, cfg_mysql_script_path):
    script_dir = cfg_mysql_script_path['populate_dir']
    if feature_id not in get_feature_ids():
        print(color_message("Unsupported feature %d queued, abort" % feature_id, 'WARNING'))
        return
    feature_info = get_feature_dict(feature_id)
    print "Populate feature %s: %s" % (feature_id, feature_info["name"])
    if feature_info['extension'] == '.sql':
        conn = open_sql_connection(cfg_mysql)
        script_path = script_dir + '/' + feature_info['filename'] + feature_info['extension']
        to_be_replaced = ['moocdb',
                          'START_DATE_PLACEHOLDER',
                          'CURRENT_DATE_PLACEHOLDER',
                          'NUM_WEEKS_PLACEHOLDER']
        replace_by = [cfg_mysql['database'],
                      cfg_course['start_date'],
                      cfg_course['current_date'],
                      str(cfg_course['number_of_weeks'])]
        run_sql_script(conn=conn,
                       script_path=script_path,
                       to_be_replaced=to_be_replaced,
                       replace_by=replace_by,
                       timeout=cfg_pipeline['timeout'])
        close_sql_connection(conn)
    elif feature_info['extension'] == '.py':
        conn = open_sql_connection(cfg_mysql)
        conn2 = open_sql_connection(cfg_mysql)
        run_python_script(script_name=feature_info['filename'] + '.py',
                          script_dir=script_dir,
                          conn=conn,
                          conn2=conn2,
                          import_dir=cfg_mysql_script_path['MLQ_dir'],
                          db_name=cfg_mysql['database'],
                          start_date=cfg_course['start_date'],
                          current_date=cfg_course['current_date'],
                          num_weeks=cfg_course['number_of_weeks'],
                          timeout=cfg_pipeline['timeout'])
        close_sql_connection(conn)
        close_sql_connection(conn2)
