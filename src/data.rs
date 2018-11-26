
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LabeledData<TFeature, TLabel> {
    indices: Vec<usize>,
    values: Vec<TFeature>,
    label: TLabel,
    is_sparse: bool,
}


impl<TFeature, TLabel> LabeledData<TFeature, TLabel>
where TFeature: Clone, TLabel: Clone {
    pub fn new(
        indices: Vec<usize>,
        values: Vec<TFeature>,
        label: TLabel,
        is_sparse: bool,
    ) -> LabeledData<TFeature, TLabel> {
        return LabeledData {
            indices: indices,
            values: values,
            label: label,
            is_sparse: is_sparse,
        }
    }

    pub fn get_size(&self) -> usize {
        self.indices.len()
    }

    pub fn get_value_at(&self, k: usize) -> Option<(usize, TFeature)> {
        if k >= self.get_size() {
            return None;
        }
        if self.is_sparse {
            Some((self.indices[k], self.values[k].clone()))
        } else {
            Some((k, self.values[k].clone()))
        }
    }

    pub fn get_label(&self) -> TLabel {
        self.label.clone()
    }

    pub fn get_position(&self, index: usize) -> usize {
        let mut position = index;
        if self.is_sparse {
            let mut left = 0;
            let mut right = self.indices.len();
            while left + 1 < right {
                let mid = (left + right) / 2;
                if self.indices[mid] <= index {
                    left = mid;
                } else {
                    right = mid;
                }
            }
            if self.indices[left] < index {
                left += 1; 
            }
            position = left;
        }
        position
    }

    pub fn into(self) -> (Vec<usize>, Vec<TFeature>, TLabel, bool) {
        (self.indices, self.values, self.label, self.is_sparse)
    }
}


#[cfg(test)]
mod tests {
    use super::super::Data;
    use super::LabeledData;

    #[test]
    fn test_sparse_labeled_data() {
        let indices = vec!(1, 20, 33);
        let feature = vec!(1.0, 2.0, 3.0);
        let label: u8 = 0;
        let data = LabeledData::new(indices, feature, label, true);
        assert_eq!(data.get_value_at(2), (33, 3.0));
        assert_eq!(data.get_position(15), 1);
        assert_eq!(&data.get_label(), 0u8);
    }

    #[test]
    fn test_labeled_data() {
        let feature = vec!(1.0, 2.0, 3.0);
        let label: u8 = 0;
        let data = LabeledData::new(vec![0; 3], feature, label, false);
        assert_eq!(&data.get_value_at(2), 3.0);
        assert_eq!(&data.get_label(), 0u8);
    }
}