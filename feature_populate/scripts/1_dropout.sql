-- Takes 12 seconds
-- Created on June 30, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 1: has the student dropped out (binary, that's what we try to predict)
-- Edited by Colin Taylor on Nov 27, 2013 to include missing last submission id
-- Meant to be run after user_dropout_populate_dropout_week.sql is run

-- TRUNCATE TABLE `moocdb`.user_long_feature;
-- ALTER TABLE `moocdb`.user_long_feature AUTO_INCREMENT = 1;

DROP PROCEDURE IF EXISTS `moocdb`.compute_dropout;


CREATE PROCEDURE `moocdb`.compute_dropout()
BEGIN

  DECLARE v_max INT UNSIGNED DEFAULT 'NUM_WEEKS_PLACEHOLDER';
  DECLARE v_counter INT UNSIGNED DEFAULT 0;

  SET @current_date = CAST('CURRENT_DATE_PLACEHOLDER' AS DATETIME);
  
  WHILE v_counter < v_max DO

      INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value, date_of_extraction)
      SELECT 1, user_dropout.user_id, v_counter, 0, @current_date
      FROM `moocdb`.user_dropout AS user_dropout
      WHERE user_dropout.dropout_week <= v_counter
        AND user_dropout.dropout_week IS NOT NULL;

      INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value, date_of_extraction)
      SELECT 1, user_dropout.user_id, v_counter, 1, @current_date
      FROM `moocdb`.user_dropout AS user_dropout
      WHERE user_dropout.dropout_week > v_counter
        AND user_dropout.dropout_week  IS NOT NULL;



      SET v_counter=v_counter+1;
  END WHILE;
END;


CALL `moocdb`.compute_dropout();
