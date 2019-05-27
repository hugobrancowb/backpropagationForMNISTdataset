#!/bin/bash
#
# Script para compilar codigo C
#
# Usage: ./build.sh

#astyle -A1 -s4 -k3 -xj -SCNeUpv final
gcc final.c -o final.x -Wall -Wextra -g -std=c99 -lm -O0 -fPIC
#make final.x

echo "Codigo compilado"

while getopts ":hvftr" opt; do
  case $opt in
    h)
      scp final.c final.x hbwb@beco.poli.br:ia/rascunhos/backpropagation/. >&2
      ;;
    v)
      valgrind --leak-check=full --track-origins=yes ./final.x >&2
      ;;
    f)
      astyle -A1 -s4 -k3 -xj -SCNeUpv final.c >&2
      ;;
    t)
      time ./final.x >&2
      ;;
    r)
      rm wmap
      printf "\nArquivo \"wmap\" excluido.\n\n"
      ./final.x
      printf "Novo arquivo \"wmap\" gerado.\n\n"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

#chmod 755 xxx.sh