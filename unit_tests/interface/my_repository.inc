#if !defined(GT_REGISTER_FIELD) || !defined(GT_REGISTER_FIELDTYPES)
#error "GT_REGISTER_FIELD or GT_REGISTER_FIELDTYPES is not defined"
#endif

GT_REGISTER_FIELDTYPES(IJKDataStore, IJDataStore) // should be possible as well (VA_ARGS trick from list)

GT_REGISTER_FIELD(IJKDataStore, u)
GT_REGISTER_FIELD(IJKDataStore, w)
GT_REGISTER_FIELD(IJKDataStore, v)
GT_REGISTER_FIELD(IJDataStore, crlat)
