项目为 Aleo 矿池 Prover 测试，

aleo-pool-server 
cargo build  
 ./target/debug/aleo-pool-server --address aleo1ug3fvdn9nsyjc24vc2wf8utfw0yq94d7yykjsqv4m43qr22zkczqctewwa --port 6688 --api-port 8888 


aleo-prover
cargo build --release --features cuda
 ./target/release/aleo-prover --address aleo1ug3fvdn9nsyjc24vc2wf8utfw0yq94d7yykjsqv4m43qr22zkczqctewwa --pool localhost:6688
