#!/bin/bash

FOUND=0

for i in {1..3}
do
  echo -n Password:
  read -s password
  echo

  echo -n Repeat password:
  read -s password_repeat
  echo

  if [ "$password" = "$password_repeat" ]
  then
    (( FOUND++ ))
    break
  else
    echo "The passwords do not match. Retrying"
  fi
done

if [ $FOUND == 0 ]
then
  echo "Too many retries, exiting"
  exit 1
fi

HASHED=`echo -n $password | openssl sha1 | awk '{print $2}'`
OUTPUT="sha1::${HASHED}"
echo $OUTPUT | tee password.txt