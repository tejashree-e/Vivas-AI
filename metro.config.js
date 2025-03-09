const { mergeConfig } = require('@react-native/metro-config');

const { getDefaultConfig } = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

module.exports = mergeConfig(defaultConfig, {
  resolver: {
    sourceExts: ["jsx", "js", "ts", "tsx"],
  },
});
