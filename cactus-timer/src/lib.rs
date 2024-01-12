use std::{ time::Instant};
use colored::Colorize;

pub struct TimerInfo {
    pub msg: String,
    pub time: Instant,
}

pub fn start_timer(msg: &str) -> TimerInfo {

    //let start_info = "Start:".yellow().bold();
    // println!("{:8} {}", start_info, msg);

    TimerInfo{msg: msg.to_string(), time: Instant::now()}   
}

pub fn end_timer(time: &TimerInfo, msg: &str){

    let final_time = time.time.elapsed();
    let final_time = {
        let secs = final_time.as_secs();
        let millis = final_time.subsec_millis();
        let micros = final_time.subsec_micros() % 1000;
        let nanos = final_time.subsec_nanos() % 1000;
        if secs != 0 {
            format!("{}.{:0>3}s", secs, millis).bold()
        } else if millis > 0 {
            format!("{}.{:0>3}ms", millis, micros).bold()
        } else if micros > 0 {
            format!("{}.{:0>3}µs", micros, nanos).bold()
        } else {
            format!("{}ns", final_time.subsec_nanos()).bold()
        }
    };

    let end_info = "End:".green().bold();
    let message = format!("{} {}", time.msg, msg);

    println!("{:8} {} {}", end_info, message, final_time);

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
       let time = start_timer("hello world");
       end_timer(time, "i am here");
    }
}
