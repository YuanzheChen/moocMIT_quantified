# /usr/bin/env python
import sys
import os
import argparse
import subprocess
import tempfile
import getpass
import yaml
import MySQLdb

# This script serves as an entry-point to config and run MLQ automatically
# to populate the feature values of courses into the moocdb database (or several databases).
# TODO: course section of MLQ config should be retrieved from MOOCdb databases, but not hardcoded
# TODO: Extend MLC and MLQ autorun to work with multiple databases.

MOOCDB_CONFIG_SUB_STRUCTURE = {
    "MOOCdb": {
        "database": None,
        "work_dir": None,
        "MLQ_folder": None
    },
    "mysql": {
        "host": None,
        "password": None,
        "port": None,
        "user": None
    },
    "full_pipeline": {
        "MLQ": None
    },
    "MLQ_pipeline": {
        "preprocess": None,
        "timeout": None,
        "is_testing": None,
        "skip": None,
        "test": None
    },
}

DEFAULT_PIPELINE = {
        "query": False,
        "preprocess": True,
        "timeout": 1000,
        "is_testing": False,
        "skip": None,
        "test": None
}

MLQ_CONFIG_TEMPLATE = {
    "course": {
        "start_date": "2017-03-07 14:00:00",
        "number_of_weeks": 9
    },
    "mysql_script_path": {
        "MLQ_dir": None,
        "preprocess_folder": "preprocess",
        "populate_folder": "feature_populate"
    },
    "mysql": {
        "query_user": False,
        "database": None,
        "query_password": False,
        "host": None,
        "user": None,
        "query_database": False,
        "password": None,
        "port": None
    }
}


def dict_structure(d):
    # Extract out the structure of dict cfg to
    # compare with the legal structure
    if isinstance(d, dict):
        return {k: dict_structure(d[k]) for k in d}
    else:
        # Replace all non-dict values with None.
        return None


def db_config_check(sup_dict, sub_dict):
    return all(item in dict_structure(sup_dict).items()
               if not isinstance(item[1], dict)
               else db_config_check(sup_dict[item[0]], sub_dict[item[0]])
               for item in dict_structure(sub_dict).items())


def autorun():
    # Parse arguments
    parser = argparse.ArgumentParser('Auto-config and run MLQ automatically to populate'
                                     'the feature table.')
    parser.add_argument('-c', action="store", default=None, dest='db_config_path',
                        help='path to MOOCdb config file')
    parser.add_argument("-s", action="store", default='moocdb', dest='db', help='MySQL moocdb database name')
    parser.add_argument("-t", action="store", default='.', dest='MLQ_path', help='path to MLQ')
    db_config_path = parser.parse_args().db_config_path

    # MOOCdb config
    if not db_config_path:
        moocdb = parser.parse_args().db
        MLQ_path = parser.parse_args().MLQ_path
        MLQ_CONFIG_TEMPLATE['mysql']['host'] = 'localhost'
        MLQ_CONFIG_TEMPLATE['mysql']['port'] = 3306
        MLQ_CONFIG_TEMPLATE['mysql']['user'] = raw_input('Enter your username for MySQL: ')
        MLQ_CONFIG_TEMPLATE['mysql']['password'] = getpass.getpass('Enter corresponding password: ')
        MLQ_CONFIG_TEMPLATE['pipeline'] = DEFAULT_PIPELINE
        MLQ_CONFIG_TEMPLATE['pipeline']['query'] = True
    else:
        db_config_path = os.path.abspath(db_config_path)
        if not os.path.isfile(db_config_path):
            sys.exit('Specified MOOCdb config file does not exist.')
        db_cfg = yaml.safe_load(open(db_config_path))
        if not db_config_check(db_cfg, MOOCDB_CONFIG_SUB_STRUCTURE):
            sys.exit('MOOCdb config file is invalid.')
        # Exit with code 0 if not queued
        if not db_cfg['full_pipeline']['MLQ']:
            print('MLQ is not queued, container exited.')
            exit(0)
        moocdb = db_cfg['MOOCdb']['database']
        MLQ_path = os.path.join(
            db_cfg['MOOCdb']['work_dir'],
            db_cfg['MOOCdb']['MLQ_folder'] + '/'
        )
        MLQ_CONFIG_TEMPLATE['mysql']['host'] = db_cfg['mysql']['host']
        MLQ_CONFIG_TEMPLATE['mysql']['port'] = db_cfg['mysql']['port']
        MLQ_CONFIG_TEMPLATE['mysql']['user'] = db_cfg['mysql']['user']
        MLQ_CONFIG_TEMPLATE['mysql']['password'] = db_cfg['mysql']['password']
        MLQ_CONFIG_TEMPLATE['pipeline'] = db_cfg['MLQ_pipeline']

    # Check the MLQ folder
    MLQ_path = os.path.abspath(MLQ_path)
    if not os.path.isdir(MLQ_path):
        sys.exit('MLQ directory does not exist.')
    MLQ_files = [f for f in os.listdir(MLQ_path)]
    if 'full_pipe.py' not in MLQ_files:
        sys.exit("MLQ directory is not complete.")
    MLQ_exec = os.path.join(MLQ_path, 'full_pipe.py')

    # Check moocdb database exist
    db = MySQLdb.connect(host=MLQ_CONFIG_TEMPLATE['mysql']['host'],
                         port=MLQ_CONFIG_TEMPLATE['mysql']['port'],
                         user=MLQ_CONFIG_TEMPLATE['mysql']['user'],
                         passwd=MLQ_CONFIG_TEMPLATE['mysql']['password'],
                         db=moocdb)
    cursor = db.cursor()
    cursor.execute("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA " +
                   "WHERE SCHEMA_NAME = '%s'" % moocdb)
    if cursor.rowcount == 0:
        sys.exit('MOOCdb database dose not exist.')

    cursor.close()

    # Finally set the two paths after check
    if db_config_path:
        MLQ_CONFIG_TEMPLATE['pipeline']['query'] = False
    MLQ_CONFIG_TEMPLATE['mysql']['database'] = moocdb
    MLQ_CONFIG_TEMPLATE['mysql_script_path']['MLQ_dir'] = MLQ_path

    # Run MLQ as subprocess
    print("Running MLQ on database %s" % moocdb)
    config_dict = MLQ_CONFIG_TEMPLATE
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml') as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)
        p = subprocess.Popen(sys.executable + ' -u ' + MLQ_exec + ' ' + config_file.name,
                             shell=True, stderr=subprocess.PIPE, bufsize=1)
        print("Process pid: %d" % p.pid)
        with p.stderr:
            for line in iter(p.stderr.readline, ''):
                print(line)
        p.wait()

if __name__ == "__main__":
    autorun()
