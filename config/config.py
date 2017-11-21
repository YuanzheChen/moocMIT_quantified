#!/usr/bin/env python
'''
This is the parser of YAML configuration file.

Parse the YAML config file and store all configuration variables as constants.
'''
import sys
import os
import yaml
import getpass
import datetime

from feature_populate.feature_dict import USER_FEATURE_CATALOG, process_exec, process_skip


class ConfigParser(object):
    '''
    Handles parsing and processing the config YAML file and
    act as an interface for other modules to read config information.
    '''

    CONFIG_STRUCTURE = {
        "course": {
            "number_of_weeks": None,
            "start_date": None,
        },
        "pipeline": {
            "query": None,
            "preprocess": None,
            "skip": None,
            "is_testing": None,
            "test": None,
            "timeout": None
        },
        "mysql": {
            "query_user": None,
            "database": None,
            "query_password": None,
            "host": None,
            "user": None,
            "query_database": None,
            "password": None,
            "port": None
        },
        "mysql_script_path": {
            "MLQ_dir": None,
            "preprocess_folder": None,
            "populate_folder": None
        }
    }

    def __init__(self, path='./config/config.yml'):
        # Parse YAML file and check validity
        self.cfg = yaml.safe_load(open(path))
        self.validity = self.check()
        if self.validity:
            self.pre_process()

    def check(self):
        # Check whether the structure of cfg is valid
        return self.dict_structure(self.cfg) == self.CONFIG_STRUCTURE

    def dict_structure(self, d):
        # Extract out the structure of dict cfg to
        # compare with the legal structure
        if isinstance(d, dict):
            return {k: self.dict_structure(d[k]) for k in d}
        else:
            # Replace all non-dict values with None.
            return None

    def is_valid(self):
        return self.validity

    def pre_process(self):
        # Get current date:
        self.cfg['course']['current_date'] = datetime.datetime.now().isoformat()
        # pre-process the fundamental dirs
        self.cfg['mysql_script_path']['MLQ_dir'] = os.path.abspath(
            self.cfg['mysql_script_path']['MLQ_dir']
        )
        # pipeline MySQL script dirs
        self.cfg['mysql_script_path']['preprocess_dir'] = os.path.join(
            self.cfg['mysql_script_path']['MLQ_dir'],
            self.cfg['mysql_script_path']['preprocess_folder'] + '/',
            'scripts/'
        )
        self.cfg['mysql_script_path']['populate_dir'] = os.path.join(
            self.cfg['mysql_script_path']['MLQ_dir'],
            self.cfg['mysql_script_path']['populate_folder'] + '/',
            'scripts/'
        )
        self.cfg['course']['start_date'] = str(self.cfg['course']['start_date'])

    def get_course(self):
        return self.cfg['course']

    def get_mysql_script_path(self):
        return self.cfg['mysql_script_path']

    def get_or_query_mysql(self):
        cfg_mysql = self.cfg['mysql']
        if cfg_mysql['query_user']:
            cfg_mysql['user'] = raw_input('Enter your username for MySQL: ')
        if cfg_mysql['query_password']:
            cfg_mysql['password'] = getpass.getpass('Enter corresponding password of user %s: ' % cfg_mysql['user'])
        if cfg_mysql['query_database']:
            cfg_mysql['database'] = raw_input('Enter the database name: ')
        credential_list = [
            'host',
            'port',
            'user',
            'password',
            'database'
        ]
        return {k: cfg_mysql[k] for k in credential_list if k in cfg_mysql}

    def get_or_query_pipeline(self, default='y'):
        cfg_pipeline = self.cfg['pipeline']
        valid = {"yes": True, "y": True, "no": False, "n": False}
        prompt = " (default: %s) [y/n] " % default
        query = cfg_pipeline['query']
        selected_ids = []

        if query:
            # Query if run pre-process
            while True:
                sys.stdout.write("Run pre-process before feature extraction?" + prompt)
                choice = raw_input().lower()
                if default is not None and choice == '':
                    shall_preprocess = valid[default]
                    break
                elif choice in valid:
                    shall_preprocess = valid[choice]
                    break
                else:
                    sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
            cfg_pipeline['preprocess'] = shall_preprocess
            # Query if entering test mode
            while True:
                sys.stdout.write("Enter test mode?" + prompt)
                choice = raw_input().lower()
                if default is not None and choice == '':
                    is_testing = valid[default]
                    break
                elif choice in valid:
                    is_testing = valid[choice]
                    break
                else:
                    sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
            cfg_pipeline['is_testing'] = is_testing

            # Query the list of features
            print("The completeis_testing list of all features:")
            print("[ id]: %s: description" % ('{0: <50}'.format('name')))
            for fid in sorted(USER_FEATURE_CATALOG.keys()):
                print("[%s]: %s: %s" % (str(fid).zfill(3),
                                        '{0: <50}'.format(USER_FEATURE_CATALOG[fid]['name']),
                                        USER_FEATURE_CATALOG[fid]['desc']))
            print('\n')
            while True:
                try:
                    selected_ids = raw_input("Please indicate the selected features "
                                             "by entering their ids, and split them by commas:")
                    selected_ids = list(set(selected_ids.split(',')))
                    selected_ids = map(int, selected_ids)
                    selected_features = [USER_FEATURE_CATALOG[x]['name'] for x in selected_ids]
                    if selected_ids:
                        break
                except (ValueError, SyntaxError, NameError) as e:
                    print("Invalid id found, error message: %s. Please re-enter." % str(e))
                except KeyError as e:
                        print("Invalid id <%s> found. Please re-enter." % str(e))
            print('\n')
            print("Selected features: %s" % str(selected_features))
            print("These features will be %s" % ('executed' if is_testing else 'skipped'))
        # Change None to empty list
        if not cfg_pipeline['test']:
            cfg_pipeline['test'] = []
        if not cfg_pipeline['skip']:
            cfg_pipeline['skip'] = []
        # Process the test or skip list
        if cfg_pipeline['is_testing']:
            cfg_pipeline['exec'] = process_exec(selected_ids if query else cfg_pipeline['test'])
        else:
            cfg_pipeline['exec'] = process_skip(selected_ids if query else cfg_pipeline['skip'])
        print("Features %s queued" % str(cfg_pipeline['exec']))
        used_list = [
            'exec',
            'timeout',
            'preprocess'
        ]
        return {k: cfg_pipeline[k] for k in used_list if k in cfg_pipeline}
