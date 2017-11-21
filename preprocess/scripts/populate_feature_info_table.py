"""
Populate feature_info table as a dictionary of features.
To be used by MLV and MLM later to retrieve feature information.
"""
def main(**kwargs):
    """
    populates 3 fields of the moocdb.feature_info table:
        feature_id, feature_name, feature_description
    with features_dictionary values:
         key,name,desc
    DOES NOT POPULATE the value of a feature
    """
    # Parse the kwargs
    conn = kwargs['conn']
    db_name = kwargs['db_name']
    import_path = kwargs['import_dir']
    parent_conn = kwargs['parent_conn']
    # Dynamic import the required module
    try:
        import sys
        sys.path.append(import_path)
        from feature_populate.feature_dict import get_feature_ids, get_feature_dict
    except RuntimeWarning:
        pass
    # Prepare the statement
    cursor = conn.cursor()
    insertion = '''INSERT INTO `%s`.`feature_info`
                                (`feature_id`,
                                 `feature_table`,
                                 `feature_name`,
                                 `feature_description`)
                   VALUES ''' % db_name

    values = [insertion]
    for feature_id in get_feature_ids():
        feature_table = get_feature_dict(feature_id)['table']
        name = get_feature_dict(feature_id)['name']
        description = get_feature_dict(feature_id)['desc']
        values.append('''(%d, "%s", "%s", "%s"),''' % (feature_id, feature_table, name, description))
    # Remove the last comma
    values[-1] = values[-1][:-1]
    # Add the delimiter
    values.append(';')
    # Join to get a single statement
    values = "".join(values)

    # Execute the statement
    cursor.execute(values)

    # Commit and close the cursor
    conn.commit()
    cursor.close()

    # Report success to parent process
    if parent_conn:
        parent_conn.send(True)

