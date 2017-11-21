DROP TABLE if exists `moocdb`.`user_dropout`;

CREATE TABLE `moocdb`.`user_dropout` (
  `user_id` VARCHAR(50) NOT NULL,
  `last_submission_id` VARCHAR(50) NULL,
  `dropout_week` INT(2) NULL ,
  `dropout_timestamp` DATETIME NULL,
  PRIMARY KEY (`user_id`)
);
