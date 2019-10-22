use std::fs::rename;
use std::fs::remove_file;
use std::io::Write;

use REGION;
use BUCKET;
use bincode::deserialize;
use bincode::serialize;
use commons::ExampleWithScore;
use commons::io::read_all;
use commons::io::write_all;
use commons::io::create_bufwriter;
use commons::io::raw_read_all;
use commons::io::load_s3 as io_load_s3;
use commons::io::write_s3 as io_write_s3;
use commons::Model;


pub type VersionedSampleModel = (usize, Vec<ExampleWithScore>, Model);


const S3_PATH_SAMPLE:  &str = "sparrow-samples/";
const SAMPLE_FILENAME: &str = "sample.bin";
const S3_PATH_MODELS:  &str = "sparrow-models/";
const MODEL_FILENAME:  &str = "model.bin";
const S3_PATH_ASSIGNS: &str = "sparrow-assigns/";
const ASSIGN_FILENAME: &str = "assign.bin";


// For gatherer

pub fn write_sample_local(
    new_sample: Vec<ExampleWithScore>,
    model: Model,
    version: usize,
    _exp_name: &str,
) {
    let filename = SAMPLE_FILENAME.to_string() + "_WRITING";
    let data: VersionedSampleModel = (version, new_sample, model);
    write_all(&filename, &serialize(&data).unwrap())
        .expect("Failed to write the sample set to file");
    rename(filename, SAMPLE_FILENAME.to_string()).unwrap();
}


pub fn write_sample_s3(
    new_sample: Vec<ExampleWithScore>,
    model: Model,
    version: usize,
    exp_name: &str,
) {
    let data: VersionedSampleModel = (version, new_sample, model);
    debug!("sampler, start, write new sample to s3, {}", version);
    let s3_path = format!("{}/{}", exp_name, S3_PATH_SAMPLE);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), SAMPLE_FILENAME, &serialize(&data).unwrap());
    debug!("sampler, finished, write new sample to s3, {}", version);
    let filename = SAMPLE_FILENAME.to_string() + "_WRITING";
    write_all(&filename, &serialize(&data).unwrap())
        .expect(format!("Failed to write the sample set to file, {}", version).as_str());
    rename(filename, format!("{}_{}", SAMPLE_FILENAME, version)).unwrap();
}


// For loader

pub fn load_sample_local(last_version: usize, _exp_name: &str) -> Option<VersionedSampleModel> {
    let ori_filename = SAMPLE_FILENAME.to_string();
    let filename = ori_filename.clone() + "_READING";
    if rename(ori_filename, filename.clone()).is_ok() {
        let (version, sample, model): VersionedSampleModel =
            deserialize(read_all(&filename).as_ref()).unwrap();
        if version > last_version {
            remove_file(filename).unwrap();
            return Some((version, sample, model));
        }
    }
    None
}


pub fn load_sample_s3(last_version: usize, exp_name: &str) -> Option<VersionedSampleModel> {
    // debug!("scanner, start, download sample from s3");
    let s3_path = format!("{}/{}", exp_name, S3_PATH_SAMPLE);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), SAMPLE_FILENAME);
    if ret.is_none() {
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        let (version, sample, model) = deserialize(&data).unwrap();
        if version > last_version {
            return Some((version, sample, model));
        }
        debug!("scanner, finished, download sample from s3, remote sample is old, {}, {}",
               version, last_version);
    } else {
        debug!("scanner, failed, download sample from s3, err {}", code);
    }
    None
}

// read/write model.json

pub fn write_model(model: &Model, timestamp: f32, save_process: bool) {
    let json = serde_json::to_string(&(timestamp, model.size(), model)).expect(
        "Local model cannot be serialized."
    );
    let filename = {
        if save_process {
            format!("models/model_{}-v{}.json", model.size(), model.size())
        } else {
            "model.json".to_string()
        }
    };
    create_bufwriter(&filename).write(json.as_ref()).unwrap();
}


pub fn read_model() -> (f32, usize, Model) {
    serde_json::from_str(&raw_read_all(&"model.json".to_string()))
            .expect(&format!("Cannot parse the model in `model.json`"))
}


// Worker download models
pub fn download_model(exp_name: &String) -> Option<(Model, String, f32, f32)> {
    // debug!("sampler, start, download model");
    let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME);
    // debug!("sampler, finished, download model");
    if ret.is_none() {
        debug!("sample, download model, failed");
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        // debug!("sample, download model, succeed");
        Some(deserialize(&data).unwrap())
    } else {
        debug!("sample, download model, failed with return code {}", code);
        None
    }
}


pub fn download_assignments(exp_name: &String) -> Option<Vec<Option<usize>>> {
    let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
    let ret = io_load_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME);
    // debug!("model sync, finished, download assignments");
    if ret.is_none() {
        // debug!("model sync, download assignments, failed");
        return None;
    }
    let (data, code) = ret.unwrap();
    if code == 200 {
        // debug!("model sync, download assignments, succeed");
        Some(deserialize(&data).unwrap())
    } else {
        debug!("model sync, download assignments, failed with return code {}", code);
        None
    }
}


// Server upload models
pub fn upload_model(
    model: &Model, sig: &String, gamma: f32, root_gamma: f32, exp_name: &String,
) -> bool {
    let data: (Model, String, f32, f32) = (model.clone(), sig.clone(), gamma, root_gamma);
    let s3_path = format!("{}/{}", exp_name, S3_PATH_MODELS);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), MODEL_FILENAME, &serialize(&data).unwrap())
}


// Server upload assignments
pub fn upload_assignments(worker_assign: &Vec<Option<usize>>, exp_name: &String) -> bool {
    let data = worker_assign;
    let s3_path = format!("{}/{}", exp_name, S3_PATH_ASSIGNS);
    io_write_s3(REGION, BUCKET, s3_path.as_str(), ASSIGN_FILENAME, &serialize(&data).unwrap())
}