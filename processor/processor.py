from __future__ import print_function

import MySQLdb
import MySQLdb.cursors as cursors
import imp
import re
import sqlparse
import multiprocessing
import time
import traceback


# Process class to handle child process's exception message
class Process(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class ScriptExecutionException(Exception):
    pass


class TimeoutException(Exception):
    pass


def color_message(message, _type):
    color_unicode = {
        'OK': '\033[92m',
        'WARNING': '\033[93m',
        'FAIL': '\033[91m',
        'INFO': '\033[94m',
        'ERROR': '\033[31m',
        'END': '\033[0m',
    }
    if _type not in ('OK', 'WARNING', 'FAIL', 'INFO', 'ERROR'):
        return message
    else:
        return color_unicode[_type] + message + color_unicode['END']


def open_sql_connection(cfg_mysql):
    return MySQLdb.connect(host=cfg_mysql['host'],
                           port=cfg_mysql['port'],
                           user=cfg_mysql['user'],
                           passwd=cfg_mysql['password'],
                           db=cfg_mysql['database'],
                           cursorclass=cursors.SSCursor)


def close_sql_connection(connection):
    connection.close()


def execute_sql(connection, command, parent_conn=None):
    """Execute sql command, which is a sequence of SQL statements
    separated by ";" and possibly "\n"
    """
    # split commands by \n
    commands = command.split("\n")
    # remove comments
    commands = [x for x in commands if x.lstrip()[:2] != '--']
    # remove extra \n and \r as they are the newline character
    commands = [re.sub('\n', '', x) for x in commands if x.lstrip() != '\n']
    commands = [re.sub('\r', '', x) for x in commands if x.lstrip() != '\r']
    # remove white spaces
    commands = [x.strip() for x in commands if x.strip() != '']
    # re-combine multi-line commands
    multi_line_commands = []
    multi_line_command = ''
    for single_line_command in commands:
        multi_line_command += single_line_command + ' '
        if multi_line_command.endswith('; '):
            multi_line_commands.append(multi_line_command.rstrip())
            multi_line_command = ''
    commands = multi_line_commands
    # re-combine stored procedures and functions with delimiter ';' inside
    procedure_commands = []
    procedure_command = ''
    procedure_flag = False
    for simple_command in commands:
        is_start = (simple_command.upper().startswith('CREATE FUNCTION')
                    or
                    simple_command.upper().startswith('CREATE PROCEDURE'))
        is_end = simple_command.upper().endswith('END;')
        if not procedure_flag and is_start and is_end:
            # procedural command in one line
            procedure_commands.append(simple_command)
        if not procedure_flag and is_start:
            # start of multi-line procedural commands
            procedure_flag = True
            procedure_command = simple_command
        elif not procedure_command:
            # simple command
            procedure_commands.append(simple_command)
        elif procedure_commands and not is_end:
            # inside multi-line procedural command
            procedure_command += simple_command + ' '
        elif procedure_commands:
            # end of multi-line procedure command
            procedure_command += simple_command
            procedure_commands.append(procedure_command)
            procedure_command = ''
            procedure_flag = False
    commands = procedure_commands

    for command in commands:
        statements = sqlparse.split(command)
        for statement in statements:
            cur = connection.cursor()
            # make sure actually does something
            if sqlparse.parse(statement):
                cur.execute(statement)
            cur.close()
    connection.commit()

    if parent_conn:
        parent_conn.send(True)


# Not fully understand its usage
# TODO: understand it and probably change it in the future
def block_sql_command(conn, cursor, command, data, block_size):
    last_block = False
    current_offset = 0
    while not last_block:
        if current_offset + block_size < len(data):
            block = data[current_offset:current_offset + block_size]
        else:
            block = data[current_offset:]
            last_block = True
        if block:
            cursor.executemany(command, block)
            conn.commit()
            current_offset += block_size


def replace_words_in_script(file_name, to_be_replaced, replace_by):
    script_file = open(file_name, 'r').read()
    if len(to_be_replaced) != len(replace_by):
        raise ValueError("The sizes of to_be_replaced and replace_by lists must be the same")
    else:
        for i in range(len(to_be_replaced)):
            script_file = re.subn(re.escape(to_be_replaced[i]), replace_by[i], script_file)[0]
    return script_file


def run_sql_script(conn, script_path, to_be_replaced, replace_by, timeout):
    success = True
    is_timeout = False
    error_message = []
    begin_time = time.time()
    print("Executing MySQL script:  {:.40}".format(script_path.rsplit('/', 1)[-1]), end='')
    try:
        commands = replace_words_in_script(script_path, to_be_replaced, replace_by)
        conn1_rcv, conn2_send = multiprocessing.Pipe(False)
        subproc = Process(target=execute_sql, args=(conn, commands, conn2_send))
        subproc.start()
        subproc.join(timeout)
        subproc.terminate()
        if subproc.exception:
            error, traceback_message = subproc.exception
            error_message.append('Error: ' + str(error) + '\n' + str(traceback_message))
            raise ScriptExecutionException
        if not conn1_rcv.poll():
            raise TimeoutException()
        else:
            success = conn1_rcv.recv()
    except (IOError, SyntaxError, ValueError) as e:
        success = False
        error_message.append('Error: ' + str(e) + ": " + script_path.rsplit('/', 1)[-1] + '\n')
    except ScriptExecutionException:
        success = False
    except TimeoutException:
        is_timeout = True
        error_message.append('Error: Script ran for >%d seconds' % timeout)
    if not success:
        status = color_message('     Failed        ', 'FAIL')
    elif is_timeout:
        status = color_message('     Stopped       ', 'WARNING')
    else:
        status = color_message('     Succeeded     ', 'OK')
    print(status, end='')
    end_time = time.time()
    print(color_message("Elapsed time = %.2fs" % (end_time - begin_time), 'INFO'))
    if error_message:
        error_message = ''.join(['[%d]: ' % (i + 1) + m for i, m in enumerate(error_message)])
        print(color_message('Error message: ', 'ERROR'))
        print(color_message(error_message, 'ERROR'))


def run_python_script(script_name, script_dir, conn, conn2, import_dir, db_name, start_date,
                      current_date, num_weeks, timeout):
    success = True
    is_timeout = False
    error_message = []
    begin_time = time.time()
    print("Executing Python script: {:.40}".format(script_name), end='')
    try:
        script = imp.load_source(script_name[-3:], script_dir + script_name)
        kwargs = {
            'conn': conn,
            'conn2': conn2,
            'import_dir': import_dir,
            'script_dir': script_dir,
            'file_name': script_name,
            'db_name': db_name,
            'start_date': start_date,
            'current_date': current_date,
            'num_weeks': num_weeks,
            'timeout': timeout,
            'parent_conn': None,
        }
        conn1_rcv, conn2_send = multiprocessing.Pipe(False)
        kwargs['parent_conn'] = conn2_send
        subproc = Process(target=script.main, kwargs=kwargs)
        subproc.start()
        subproc.join(timeout)
        subproc.terminate()
        if subproc.exception:
            error, traceback_message = subproc.exception
            error_message.append('Error: ' + str(error) + '\n' + str(traceback_message))
            raise ScriptExecutionException
        if not conn1_rcv.poll():
            raise TimeoutException()
        else:
            success = conn1_rcv.recv()
    except (IOError, SyntaxError) as e:
        success = False
        error_message.append('Error: ' + str(e) + ": " + script_name + '\n')
    except ScriptExecutionException:
        success = False
    except TimeoutException:
        is_timeout = True
        error_message.append('Error: Script ran for >%d seconds' % timeout)
    if not success:
        status = color_message('     Failed        ', 'FAIL')
    elif is_timeout:
        status = color_message('     Stopped       ', 'WARNING')
    else:
        status = color_message('     Succeeded     ', 'OK')
    print(status, end='')
    end_time = time.time()
    print(color_message("Elapsed time = %.2fs" % (end_time - begin_time), 'INFO'))
    if error_message:
        error_message = ''.join(['[%d]: ' % (i+1) + m for i, m in enumerate(error_message)])
        print(color_message('Error message: ', 'ERROR'))
        print(color_message(error_message, 'ERROR'))

