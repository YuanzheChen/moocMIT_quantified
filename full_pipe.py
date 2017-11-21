#!/usr/bin/env python
"""
This is the complete moocdb/MOOC-Learner-Quantified pipeline, pre-processing followed by feature population.
After configuration, run in the command line using:
python full_pipe.py
"""
import sys
import time
import warnings
# Hide all warnings for better output alignment
warnings.filterwarnings('ignore')

from config import config
from feature_populate import feature_populate
from preprocess import preprocess
from processor.processor import color_message


def run_preprocess(cfg_mysql, cfg_pipeline, cfg_course, cfg_mysql_script_path):
    print "###### Step 1: Pre-processing database"
    preprocess.preprocess(cfg_mysql, cfg_pipeline, cfg_course, cfg_mysql_script_path)
    print "Done"


def run_feature_population(cfg_mysql, cfg_pipeline, cfg_course, cfg_mysql_script_path):
    print "###### Step 2: Population features"
    feature_populate.populate(cfg_mysql, cfg_pipeline, cfg_course, cfg_mysql_script_path)
    print "Done"


def main():
    """Performs pre-processing and feature population.
    Current pre-processing of full_pipe result entails:
      1. Creating user_info table and populating the
      dropout and last submission date for users in
      2. Creating the feature_info table to be used by MLV and MLM
      3. Creating tables for features of different types of objects
      to be placed in which are named as:
          [type of object]_features
      The possible types of objects include: user, video, forum, submission, ...

    """
    cfg_mysql = cfg.get_or_query_mysql()
    cfg_pipeline = cfg.get_or_query_pipeline()
    cfg_course = cfg.get_course()
    cfg_mysql_script_path = cfg.get_mysql_script_path()

    print("MySQL database Name: %s" % cfg_mysql['database'])
    print("Current date: %s" % cfg_course['current_date'])

    if cfg_pipeline['preprocess']:
        run_preprocess(cfg_mysql, cfg_pipeline, cfg_course, cfg_mysql_script_path)
    run_feature_population(cfg_mysql, cfg_pipeline, cfg_course, cfg_mysql_script_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        cfg = config.ConfigParser()
    else:
        cfg = config.ConfigParser(sys.argv[1])
    if not cfg.is_valid():
        sys.exit("Config file is invalid.")
    start_time = time.time()
    main()
    end_time = time.time()
    print('')
    print(color_message('All queued jobs have been executed', 'INFO'))
    print(color_message('Elapsed time in seconds: %fs' % (end_time - start_time), 'INFO'))
