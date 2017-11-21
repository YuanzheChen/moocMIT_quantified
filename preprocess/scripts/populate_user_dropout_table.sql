-- Populate the user_id attribute of the user_dropout table using submissions table user_id's

INSERT INTO `moocdb`.user_dropout (user_id)
SELECT DISTINCT user_id from submissions
ORDER BY user_id;

-- Adapted by Alex Huang (alhuang10@gmail.com) from a script by Franck Dernoncourt for ALFA, MIT lab: franck.dernoncourt@gmail.com

-- First create index:  (takes 250 seconds to run)
-- ALTER TABLE submissions
-- ADD INDEX `user-timestamp_idx` (`user_id` ASC, `submission_timestamp` ASC) ;

-- sql_mode must be set to '' or else script will throw error

SET @original_sql_mode = @@SESSION.sql_mode;

SET SESSION sql_mode = 'NO_AUTO_CREATE_USER';

UPDATE user_dropout
	SET user_dropout.last_submission_id = (
		SELECT
			submissions.submission_id
		FROM
			submissions
		WHERE
			user_dropout.user_id = submissions.user_id
				AND submissions.submission_timestamp = (
                            SELECT
                                MAX(submissions.submission_timestamp)
                            FROM
                                submissions
                            WHERE
                                user_dropout.user_id = submissions.user_id
                        )
		GROUP BY submissions.user_id
	);

SET SESSION sql_mode = @original_sql_mode;

-- Takes 4 seconds to execute
-- Created on Jun 30, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Edited by Colin Taylor on Nov 27, 2013 to
-- Meant to be run after create_dropout_feature_values.sql and user_dropout_populate_last_submission_id.sql
-- Doc by UM, Oct 2016:
-- Looks for user's last submission (recorded in user_dropout table), finds it also in the submissions table
-- and uses its timestamp (in submissions table) to record the user_dropout.dropout_week/timestamp.
-- This file has placeholders for dbname (moocdb) and startDate that caller should replace.

set @start_date = 'START_DATE_PLACEHOLDER';

--rollback if exists
SET @exist := (SELECT count(*) FROM INFORMATION_SCHEMA.COLUMNS
          WHERE TABLE_SCHEMA='moocdb' AND TABLE_NAME='user_dropout' AND COLUMN_NAME='dropout_week' );
set @sqlstmt := if( @exist > 0, 'ALTER TABLE `moocdb`.`user_dropout` DROP COlUMN `dropout_week`', 'select * from `moocdb`.`user_dropout` where 1=0');
PREPARE stmt FROM @sqlstmt;
EXECUTE stmt;

ALTER TABLE `moocdb`.`user_dropout`
ADD COLUMN `dropout_week` INT(2) NULL ;

--rollback if exists
SET @exist := (SELECT count(*) FROM INFORMATION_SCHEMA.COLUMNS
          WHERE TABLE_SCHEMA='moocdb' AND TABLE_NAME='user_dropout' AND COLUMN_NAME='dropout_timestamp' );
set @sqlstmt := if( @exist > 0, 'ALTER TABLE `moocdb`.`user_dropout` DROP COlUMN `dropout_timestamp`', 'select * from `moocdb`.`user_dropout` where 1=0');
PREPARE stmt FROM @sqlstmt;
EXECUTE stmt;

ALTER TABLE `moocdb`.`user_dropout`
ADD COLUMN `dropout_timestamp` DATETIME NULL ;

-- Match the user_dropout table user_dropout_populate_last_submission_id to submissions table submission_id
-- Calculate from submissions.submission_timestamp the week or copy timestamp (next sql cmd block)
-- Write it to user_dropout.dropout_week or user_dropout.dropout_timestamp
-- Takes 2 seconds to execute
UPDATE `moocdb`.`user_dropout`
SET user_dropout.dropout_week = (
    SELECT FLOOR((UNIX_TIMESTAMP(submissions.submission_timestamp) - UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) + 1 AS week
    FROM `moocdb`.`submissions` AS submissions
    WHERE submissions.submission_id = user_dropout.last_submission_id
)
;

-- Takes 2 seconds to execute
UPDATE `moocdb`.`user_dropout`
SET user_dropout.dropout_timestamp = (
    SELECT submissions.submission_timestamp
    FROM `moocdb`.`submissions` AS submissions
    WHERE submissions.submission_id = user_dropout.last_submission_id
)
;
