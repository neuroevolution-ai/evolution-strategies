#!/bin/bash

FOUND=0

for i in {1..3}
do
  echo -n "Enter the password you want for the Jupyter Lab Webinterface: "
  read -s password
  echo

  echo -n "Repeat the password: "
  read -s password_repeat
  echo

  if [ "$password" = "$password_repeat" ]
  then
    (( FOUND++ ))
    break
  else
    echo "The passwords do not match. Please retry."
  fi
done

if [ $FOUND == 0 ]
then
  echo "Too many retries, exiting.1"
  exit 1
fi

HASHED=`echo -n $password | openssl sha1 | awk '{print $2}'`
OUTPUT="sha1::${HASHED}"
echo "Password set to hashed_password.txt. You can build the Dockerfile now!"
echo $OUTPUT > hashed_password.txt