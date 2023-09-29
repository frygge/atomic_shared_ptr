#!/bin/bash

FILEIN=${1}
PREFIX=${2}

JOIN=$(which join)
SORT=$(which sort)

OPS=()
LIBS=()
CONTS=()

readarray -t BUFFER < <(sed -r 's/[[:blank:]]+/-/' ${FILEIN})

# parse the file and create individual files per experiment (operation/library
# combination) with a join key in the first row
echo "Phase 1..."
echo "... parsing file"
for L in "${BUFFER[@]}"; do
    A=(${L})
    case ${A[0]} in
        "===-operation:")
            OP=${A[1]}
            OPS+=(${OP})
            FILEOUT=${PREFIX}-${OP}-${LIB}-${CONT}.txt
            ;;

        "===-contention:")
            if [ "${A[1]}" == "true" ]; then
                CONT="contention"
            else
                CONT="no_contention"
            fi
            CONTS+=(${CONT})
            ;;

        "===-library:")
            LIB=${A[1]}
            LIBS+=(${LIB})
            FILEOUT=${PREFIX}-${OP}-${LIB}-${CONT}.txt
            ;;

        "===-lock_free:")
            ;;

        "vars-threads")
            rm ${FILEOUT} 2>/dev/null
            echo "... writing ${FILEOUT}"
            ;;

        "")
            ;;

        *)
            echo "${L}" >> ${FILEOUT}
            ;;
    esac
done

# join files for common graphs (i.e. different library but same operation and contention)
echo "Phase 2..."
OPS=($(echo ${OPS[@]} | tr ' ' '\n' | sort -u | tr '\n' ' '))
CONTS=($(echo ${CONTS[@]} | tr ' ' '\n' | sort -u | tr '\n' ' '))
for OP in "${OPS[@]}"; do
    for CONT in "${CONTS[@]}"; do
        readarray -t FILES < <(ls ${PREFIX}-${OP}-*-${CONT}.txt)

        IFS='-' read -r -a L <<< "${FILES[0]}"

        CMD=(cat ${FILES[0]})
        LIBS=(${L[2]})
        for(( i = 1; i < ${#FILES[@]}; i++ )); do
            IFS='-' read -r -a L <<< "${FILES[$i]}"

            CMD+=("|" ${JOIN} - ${FILES[$i]})
            LIBS+=(${L[2]})
        done

        echo "... generating ${PREFIX}-${OP}-${CONT}.txt"

        echo vars threads ${LIBS[@]} > ${PREFIX}-${OP}-${CONT}.txt
        $(eval ${CMD[@]} | sed 's/-/ /' >> ${PREFIX}-${OP}-${CONT}.txt)

        #rm ${FILES[@]}
    done
done
