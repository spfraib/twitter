func() {
	local i=$1
	mkdir d${i};  tar -C d${i} -ixf ${i}.tar; cd d${i};  tar -zcf ../d${i}.tar *; cd -; rm -rf d${i}; echo "Done ${i}"; 
}

#for v in `seq 100 999`; do func $v done

func $1

